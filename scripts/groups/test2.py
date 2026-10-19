"""Test 2 (exploratory, oracle groups): what are MNO measurements made of, and would group constraints help?

MNO is repeated on the saved maps of the frozen B0 release run (data/tcaff_baseline_release) for
every STEP-th step of all 12 directed pairs. Check-6 variants change only frame alignment, so their
maps equal B0's. Configurations:
  B0, G2            release gate (alpha=1) / relaxed gate (alpha=2, wh_scale_diff 1.7, h_diff 0.2)
  +S  structure oracle: for candidates u=(a,b), v=(a',b') with all four objects in GT groups,
      M_uv = C_uv = 0 when [g(a)=g(a')] != [g(b)=g(b')]  (a group may not be split or merged)
  +I  group-identity oracle (upper bound of a two-stage hybrid with perfect group matching):
      candidate (a,b) removed when a and b lie in different GT groups
GT group of a map object (evaluation/oracle only): nearest objects.txt box within 0.5 m XY and
0.08 <= z < 0.9 m in the mocap frame; its single-linkage (0.7 m) cluster of boxes.
Usage: python -I test2.py OUT_DIR [STEP] [WORKERS]
"""
import json
import os
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage

LAB = Path(os.environ["LAB_ROOT"])
OUT = Path(sys.argv[1])
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 14
os.environ["TCAFF_MOT_DATASET"] = str(LAB / "data/tcaff_mot_data")
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff")]
CONFIGS = ["B0", "B0+S", "B0+I", "G2", "G2+S", "G2+I"]
G2 = {"wh_scale_diff": 1.7, "h_diff": 0.2}

STATE = {}


def init():
    import clipperpy
    import tcaff_baseline_audit as audit
    from tcaff.realign.frame_aligner import FrameAligner
    from tcaff.tcaff.tcaff_manager import TCAFFManager
    params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
    STATE.update(audit=audit, params=params, poses=audit.load_poses(params), mode=None, ga=None, gb=None)
    boxes = np.loadtxt(LAB / "data/tcaff_mot_data/objects.txt", delimiter=",")
    STATE["boxes"], STATE["box_group"] = boxes, fcluster(linkage(boxes, "single"), 0.7, "distance") - 1
    orig_put = TCAFFManager.get_putative_assoc

    def put(self, map1, map2):
        A = orig_put(self, map1, map2)
        if STATE["mode"] == "I" and len(A):
            ga, gb = STATE["ga"][A[:, 0]], STATE["gb"][A[:, 1]]
            A = A[~((ga >= 0) & (gb >= 0) & (ga != gb))]
        return A

    def realign(self, detections1, detections2, downweight_nodes=None, putative_assoc=None):
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma, iparams.epsilon = self.sigma, self.epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        cparams = clipperpy.Params()
        clipper = clipperpy.CLIPPER(invariant, cparams)
        A = putative_assoc.astype(np.int32)
        clipper.score_pairwise_consistency(detections1.T, detections2.T, A)
        M_orig = clipper.get_affinity_matrix().copy()
        C = clipper.get_constraint_matrix().copy()
        if STATE["mode"] == "S":
            ga, gb = STATE["ga"][A[:, 0]], STATE["gb"][A[:, 1]]
            ing = (ga >= 0) & (gb >= 0)
            same_a = ga[:, None] == ga[None, :]
            same_b = gb[:, None] == gb[None, :]
            bad = ing[:, None] & ing[None, :] & (same_a != same_b)
            M_orig[bad] = 0.0
            C[bad] = 0
        M = M_orig.copy()
        for idx, downweight in (downweight_nodes or []):
            rows, cols = np.meshgrid(idx, idx, indexing="ij")
            if len(rows) and len(cols):
                M[rows, cols] *= downweight
        clipper = clipperpy.CLIPPER(invariant, cparams)
        clipper.set_matrix_data(M=M, C=C)
        clipper.solve()
        nodes = clipper.get_solution().nodes
        Ain = np.zeros((len(nodes), 2)).astype(np.int64)
        for i in range(len(nodes)):
            Ain[i, :] = A[nodes[i], :]
        u = clipper.get_solution().u.copy()
        for i in range(u.shape[0]):
            u[i] = u[i] if i in nodes else 0.0
        score = 0 if len(nodes) == 0 else u.T @ M_orig @ u / (u.T @ u)
        return Ain, score, nodes

    TCAFFManager.get_putative_assoc = put
    FrameAligner.realign_static_downweight = realign
    STATE["TCAFFManager"] = TCAFFManager


def labels(robot, t, M):
    p = STATE["poses"]
    T = p[(robot, "pose_gt")].T_WB(t) @ np.linalg.inv(p[(robot, "pose_estimate")].T_WB(t))
    W = (np.c_[M[:, :3], np.ones(len(M))] @ T.T)[:, :3]
    d = np.linalg.norm(W[:, None, :2] - STATE["boxes"][None], axis=2)
    nb, db = d.argmin(1), d.min(1)
    g = np.where((db < 0.5) & (W[:, 2] >= 0.08) & (W[:, 2] < 0.9), STATE["box_group"][nb], -1)
    return W, g


def job(arg):
    config, pair = arg
    from tcaff.realign.object_map import ObjectMap
    from tcaff.utils.transform import transform_2_xypsi
    audit = STATE["audit"]
    a, b = pair.split("_")
    kwargs = {k: v for k, v in STATE["params"]["tcaff"].items() if k != "ts"}
    if config.startswith("G2"):
        kwargs.update(G2)
    STATE["mode"] = config.split("+")[1] if "+" in config else None
    mgr = STATE["TCAFFManager"](**kwargs)
    data = json.loads((LAB / f"data/tcaff_baseline_release/{pair}.json").read_text())
    rows = []
    for fr in data["frames"][::STEP]:
        if not fr["map_a"] or not fr["map_b"] or fr["truth"] is None:
            continue
        A, B = np.array(fr["map_a"]), np.array(fr["map_b"])
        try:
            WA, ga = labels(a, fr["time"], A)
            WB, gb = labels(b, fr["time"], B)
        except Exception:
            continue
        STATE["ga"], STATE["gb"] = ga, gb
        mgr.get_frame_align_measurements(ObjectMap(centroids=A[:, :3], widths=A[:, 3], heights=A[:, 4], ages=A[:, 5]),
                                         ObjectMap(centroids=B[:, :3], widths=B[:, 3], heights=B[:, 4], ages=B[:, 5]))
        sols = mgr.latest_fa_solutions
        rows.append(dict(kind="step", config=config, pair=pair, step=fr["step"], n_meas=len(sols)))
        for s in sols:
            x = transform_2_xypsi(s.transform)
            t_err, y_err = audit.pose_error(x, fr["truth"])
            ij = np.asarray(s.associated_objs).reshape(-1, 2).astype(int)
            i, j = ij[:, 0], ij[:, 1]
            gi, gj = ga[i], gb[j]
            both = (gi >= 0) & (gj >= 0)
            pa = A[i, :2]
            ext = max((np.linalg.norm(p - q) for p, q in combinations(pa, 2)), default=0.0)
            rows.append(dict(kind="meas", config=config, pair=pair, step=fr["step"], t=t_err, yaw=y_err,
                             F=float(s.objective_score), n=len(ij), in_groups=int(both.sum()),
                             cross=int((both & (gi != gj)).sum()), one_side=int(((gi >= 0) ^ (gj >= 0)).sum()),
                             groups_a=len(set(gi[gi >= 0].tolist())), extent=float(ext),
                             same_place=int((np.linalg.norm(WA[i, :2] - WB[j, :2], axis=1) < 0.5).sum())))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [p.stem for p in sorted((LAB / "data/tcaff_baseline_release").glob("RR*_RR*.json"))]
    jobs = [(c, p) for c in CONFIGS for p in pairs]
    with Pool(WORKERS, initializer=init) as pool:
        for k, rows in enumerate(pool.imap_unordered(job, jobs)):
            c, p = rows[0]["config"], rows[0]["pair"]
            (OUT / f"{c}_{p}.json").write_text(json.dumps(rows))
            print(f"{k + 1}/{len(jobs)} {c} {p} {sum(r['kind'] == 'meas' for r in rows)} measurements", flush=True)
