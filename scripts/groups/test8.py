"""Test 8 (protocol: test8_protocol.json): diagnostic cluster variants KOC and KBT.
Test 7 (protocol: test7_protocol.json): test 5 with filtered on-board clusters.
Test 5 (protocol: test5_protocol.json): full TCAFF (MNO + release temporal filter) for cluster variants.
Built on test4.py; TCAFFManager.update (measurements -> faf.update) is called at every step.

Test 4 notes:

Cluster pseudo-objects are appended to each map (centre = mean of members, age = 2 x youngest member
age, so the Arun weight 1/(0.01 + a_K a_L) is about 0.25 of the members'). The size gate runs on the real
objects only; all cluster pairs are appended to its output. The affinity is built here with the release
formula (|l1 - l2| < eps -> exp(-c^2 / 2 sigma^2)); object-object entries are asserted equal to clipperpy's.
Usage: python -I test4.py OUT_DIR [STEP] [WORKERS]
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import yaml
from scipy.cluster.hierarchy import fcluster, linkage

LAB = Path(os.environ["LAB_ROOT"])
OUT = Path(sys.argv[1])
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 14
os.environ["TCAFF_MOT_DATASET"] = str(LAB / "data/tcaff_mot_data")
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff")]
CONFIGS = ["B0+KOC", "B0+KBT"]
FILT = {"KBT": (2, 0.0, 0.0), "KB": (2, 0.0, 0.0), "KB4": (4, 0.0, 0.0), "KB3RH": (3, 0.30, 0.20), "KB4RH": (4, 0.30, 0.20)}
REPEATS = 3
G2 = {"wh_scale_diff": 1.7, "h_diff": 0.2}
SIGMA_G, EPS_G = 0.30, 0.50
STATE = {}


def init():
    import clipperpy
    import tcaff_baseline_audit as audit
    from tcaff.realign.frame_aligner import FrameAligner
    from tcaff.tcaff.tcaff_manager import TCAFFManager
    params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
    STATE.update(audit=audit, params=params, poses=audit.load_poses(params), mode=None, identity=False, checked=0)
    boxes = np.loadtxt(LAB / "data/tcaff_mot_data/objects.txt", delimiter=",")
    STATE["boxes"], STATE["box_group"] = boxes, fcluster(linkage(boxes, "single"), 0.7, "distance") - 1
    orig_put = TCAFFManager.get_putative_assoc
    orig_realign = FrameAligner.realign_static_downweight

    def put(self, map1, map2):
        if STATE["mode"] is None:
            return orig_put(self, map1, map2)
        A = np.asarray(orig_put(self, STATE["real1"], STATE["real2"]), dtype=np.int64).reshape(-1, 2)
        n1, n2, k1, k2 = STATE["n1"], STATE["n2"], STATE["k1"], STATE["k2"]
        if STATE["identity"]:
            ga, gb = STATE["ga"][A[:, 0]], STATE["gb"][A[:, 1]]
            A = A[~((ga >= 0) & (gb >= 0) & (ga != gb))]
            pairs = [[n1 + K, n2 + L] for K in range(k1) for L in range(k2) if STATE["ida"][K] == STATE["idb"][L]]
        else:
            pairs = [[n1 + K, n2 + L] for K in range(k1) for L in range(k2)]
        extra = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        return np.concatenate([np.asarray(A, dtype=np.int64).reshape(-1, 2), extra])

    def realign(self, detections1, detections2, downweight_nodes=None, putative_assoc=None):
        if STATE["mode"] is None:
            return orig_realign(self, detections1, detections2, downweight_nodes, putative_assoc)
        A = putative_assoc.astype(np.int64)
        P1, P2 = np.asarray(detections1)[:, :2], np.asarray(detections2)[:, :2]
        l1 = np.linalg.norm(P1[A[:, 0]][:, None, :] - P1[A[:, 0]][None, :, :], axis=2)
        l2 = np.linalg.norm(P2[A[:, 1]][:, None, :] - P2[A[:, 1]][None, :, :], axis=2)
        c = np.abs(l1 - l2)
        cl = A[:, 0] >= STATE["n1"]
        anyc = cl[:, None] | cl[None, :]
        sig = np.where(anyc, SIGMA_G, self.sigma)
        eps = np.where(anyc, EPS_G, self.epsilon)
        M_orig = np.where(c < eps, np.exp(-0.5 * c ** 2 / sig ** 2), 0.0)
        shared = (A[:, 0][:, None] == A[:, 0][None, :]) | (A[:, 1][:, None] == A[:, 1][None, :])
        M_orig[shared] = 0.0     # clipperpy: two associations sharing a point are inconsistent (one-to-one)
        np.fill_diagonal(M_orig, 1.0)
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma, iparams.epsilon = self.sigma, self.epsilon
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        cparams = clipperpy.Params()
        if STATE["checked"] < 200:   # release formula reproduced on the object-object block
            ref = clipperpy.CLIPPER(invariant, cparams)
            ref.score_pairwise_consistency(np.asarray(detections1).T, np.asarray(detections2).T, A.astype(np.int32))
            Mref = ref.get_affinity_matrix()
            obj = ~anyc
            assert np.allclose(Mref[obj], M_orig[obj], atol=1e-9), "object-object affinity differs from clipperpy"
            STATE["checked"] += 1
        C = (M_orig > 0).astype(float)
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


def world_labels(robot, t, M):
    p = STATE["poses"]
    T_ob = p[(robot, "pose_estimate")].T_WB(t)
    T = p[(robot, "pose_gt")].T_WB(t) @ np.linalg.inv(T_ob)
    W = (np.c_[M[:, :3], np.ones(len(M))] @ T.T)[:, :3]
    d = np.linalg.norm(W[:, None, :2] - STATE["boxes"][None], axis=2)
    nb, db = d.argmin(1), d.min(1)
    g = np.where((db < 0.5) & (W[:, 2] >= 0.08) & (W[:, 2] < 0.9), STATE["box_group"][nb], -1)
    return W, g, T_ob[2, 3]


def clusters(M, g, z_robot, kind):
    """Return list of (member indices, identity)."""
    if kind == "KO":
        return [(np.flatnonzero(g == k), int(k)) for k in np.unique(g[g >= 0]) if (g == k).sum() >= 2]
    cand = np.flatnonzero((M[:, 3] <= 0.5) & (M[:, 4] >= 0.08) & (M[:, 4] <= 0.6) & (np.abs(M[:, 2] - z_robot) <= 0.35))
    if kind == "KOC":
        gc = g[cand]
        return [(cand[gc == k], int(k)) for k in np.unique(gc[gc >= 0]) if (gc == k).sum() >= 2]
    if len(cand) < 2:
        return []
    lab = fcluster(linkage(M[cand, :2], "single"), 0.7, "distance")
    out = []
    for c in np.unique(lab):
        mem = cand[lab == c]
        if not (2 <= len(mem) <= 10):
            continue
        rad = np.linalg.norm(M[mem, :2] - M[mem, :2].mean(0), axis=1).max()
        if rad > 1.0:
            continue
        nmin, rmin, hmin = FILT[kind]
        if len(mem) < nmin or rad < rmin or np.median(M[mem, 4]) < hmin:
            continue
        vals, cnt = np.unique(g[mem][g[mem] >= 0], return_counts=True)
        ident = int(vals[cnt.argmax()]) if len(vals) and cnt.max() >= 0.5 * len(mem) else -1
        if kind == "KBT" and ident < 0:
            continue
        out.append((mem, ident))
    return out


def extend(M, cl):
    rows = [M]
    for mem, _ in cl:
        rows.append(np.r_[M[mem, :3].mean(0), 0.0, 0.0, 2.0 * max(M[mem, 5].min(), 0.05)][None])
    return np.concatenate(rows)


def job(arg):
    config, pair, rep = arg
    from tcaff.realign.object_map import ObjectMap
    audit = STATE["audit"]
    a, b = pair.split("_")
    kwargs = {k: v for k, v in STATE["params"]["tcaff"].items() if k != "ts"}
    if config == "G2":
        kwargs.update(G2)
    parts = config.split("+")
    kind = parts[1] if len(parts) > 1 else None
    STATE["mode"] = kind
    STATE["identity"] = len(parts) > 2 and parts[2] == "I"
    mgr = STATE["TCAFFManager"](**kwargs)
    om = lambda X: ObjectMap(centroids=X[:, :3], widths=X[:, 3], heights=X[:, 4], ages=X[:, 5])
    data = json.loads((LAB / f"data/tcaff_baseline_release/{pair}.json").read_text())
    rows = []
    maxf = int(os.environ.get("MAXF", "0"))
    for fr in (data["frames"][:maxf] if maxf else data["frames"]):
        A = np.array(fr["map_a"], float).reshape(-1, 6)
        B = np.array(fr["map_b"], float).reshape(-1, 6)
        ca, cb = [], []
        ga, gb = np.full(len(A), -1), np.full(len(B), -1)
        if len(A) and len(B):
            try:
                _, ga, za = world_labels(a, fr["time"], A)
                _, gb, zb = world_labels(b, fr["time"], B)
                if kind:
                    ca, cb = clusters(A, ga, za, kind), clusters(B, gb, zb, kind)
            except Exception:
                pass
        STATE.update(real1=om(A), real2=om(B), n1=len(A), n2=len(B), k1=len(ca), k2=len(cb), ga=ga, gb=gb,
                     ida=[c[1] for c in ca], idb=[c[1] for c in cb])
        mgr.update(om(extend(A, ca)), om(extend(B, cb)))
        zs = mgr.latest_zs
        est = None if mgr.faf.main_tree is None else mgr.faf.main_tree.optimal.xhat.ravel()
        err = audit.pose_error(est, fr["truth"]) if est is not None and fr["truth"] is not None else None
        merr = [audit.pose_error(z.ravel(), fr["truth"]) for z in zs] if fr["truth"] is not None else []
        rows.append(dict(config=config, pair=pair, rep=rep, step=fr["step"], accepted=est is not None,
                         t=None if err is None else err[0], yaw=None if err is None else err[1],
                         n_meas=len(zs), n_meas_correct=sum(t <= 1 and y <= 5 for t, y in merr),
                         n_meas_wrong=sum(t > 2 or y > 15 for t, y in merr)))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [p.stem for p in sorted((LAB / "data/tcaff_baseline_release").glob("RR*_RR*.json"))]
    jobs = [(c, p, r) for r in range(1, REPEATS + 1) for c in CONFIGS for p in pairs]
    with Pool(WORKERS, initializer=init) as pool:
        for k, rows in enumerate(pool.imap_unordered(job, jobs)):
            c, p, r = rows[0]["config"], rows[0]["pair"], rows[0]["rep"]
            (OUT / f"{c}_{p}_rep{r}.json").write_text(json.dumps(rows))
            print(f"{k + 1}/{len(jobs)} {c} {p} rep{r} accepted {sum(x['accepted'] for x in rows)}", flush=True)
