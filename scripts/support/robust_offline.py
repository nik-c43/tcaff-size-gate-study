"""T3.0: does robust re-estimation (R3) make measurements of the correct mode more accurate? (protocol T3.0)

On the captured maps of B0 runs, every 5th step with an accepted estimate and GT, the release
MNO-CLIPPER is rerun through TCAFFManager.get_frame_align_measurements. compose_solution is wrapped
so that WLS (unchanged) and R3 are computed on the same cliques; only solutions that pass the release
filters are kept. Oracle (uses GT): WLS on correspondences with residual <= 0.5 m under the GT transform.

Usage: python -I robust_offline.py LAB_ROOT OUT_JSON CAPTURE_DIR [CAPTURE_DIR ...]
"""
import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

LAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CAPS = sys.argv[3:]
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff"), str(Path(__file__).resolve().parent)]
import patches  # noqa: E402
from h1_map_audit import object_map  # noqa: E402
from tcaff.realign.frame_aligner import FrameAligner  # noqa: E402
from tcaff.realign.wls import wls  # noqa: E402
from tcaff.tcaff.tcaff_manager import TCAFFManager  # noqa: E402
from tcaff.utils.transform import T2d_2_T3d, transform_2_xypsi, xypsi_2_transform  # noqa: E402
from tcaff.utils.transform import transform as perform_transform  # noqa: E402
from tcaff_baseline_audit import pose_error  # noqa: E402

ORACLE_R = 0.5
_orig_compose = FrameAligner.compose_solution


def _recording_compose(self, p1, p2, w):
    sol = _orig_compose(self, p1, p2, w)
    if sol.success:
        robust = patches._compose_robust(self, p1, p2, w)
        self._recorded.append((sol, robust, np.asarray(p1), np.asarray(p2), np.asarray(w).reshape(-1)))
    return sol


FrameAligner.compose_solution = _recording_compose


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def process(arg):
    rep, cap = arg
    rows, resid = [], []
    for path in sorted(Path(cap).glob("RR*_RR*.json")):
        data = json.loads(path.read_text())
        kwargs = {k: v for k, v in data["params"].items() if k != "ts"}
        for f in data["frames"]:
            if f["baseline_estimate"] is None or f["truth"] is None or f["step"] % 5:
                continue
            manager = TCAFFManager(**kwargs)
            manager.fa._recorded = []
            ma, mb = object_map(f["map_a"]), object_map(f["map_b"])
            manager.get_frame_align_measurements(ma, mb)
            kept = {id(s) for s in manager.latest_fa_solutions}
            T_truth = xypsi_2_transform(*f["truth"], dim=2)
            for sol, robust, p1, p2, w in manager.fa._recorded:
                if id(sol) not in kept:
                    continue
                z_w = transform_2_xypsi(sol.transform)
                z_r = transform_2_xypsi(robust.transform) if np.allclose(robust.transform[0, 0], robust.transform[1, 1]) else None
                r_truth = np.linalg.norm(p1 - perform_transform(T_truth, p2, stacked_axis=0), axis=1)
                good = r_truth <= ORACLE_R
                z_o = transform_2_xypsi(T2d_2_T3d(wls(p1[good], p2[good], w[good]))) if good.sum() >= 3 else None
                r_wls = np.linalg.norm(p1 - perform_transform(wls(p1, p2, w), p2, stacked_axis=0), axis=1)
                e_w = pose_error(z_w, f["truth"])
                rows.append(dict(pair=path.stem, repeat=rep, step=f["step"], n=len(p1), n_oracle=int(good.sum()),
                                 wls=e_w, r3=None if z_r is None else pose_error(z_r, f["truth"]),
                                 oracle=None if z_o is None else pose_error(z_o, f["truth"])))
                if e_w[0] <= 2.0 and e_w[1] <= 15.0:
                    resid.append((r_wls[good].tolist(), r_wls[~good].tolist()))
    return rows, resid


def ok(e):
    return e is not None and e[0] <= 1.0 and e[1] <= 5.0


if __name__ == "__main__":
    with Pool(len(CAPS)) as pool:
        parts = pool.map(process, list(enumerate(CAPS, 1)))
    rows = [r for p in parts for r in p[0]]
    resid = [r for p in parts for r in p[1]]
    inmode = [r for r in rows if r["wls"][0] <= 2.0 and r["wls"][1] <= 15.0]
    d = np.array([ok(r["r3"]) - ok(r["wls"]) for r in inmode], float)
    clusters = sorted({(r["pair"], r["repeat"]) for r in inmode})
    idx = {c: [i for i, r in enumerate(inmode) if (r["pair"], r["repeat"]) == c] for c in clusters}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        pick = [i for c in rng.choice(len(clusters), len(clusters)) for i in idx[clusters[c]]]
        boot.append(d[pick].mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])

    def share(key):
        return float(np.mean([ok(r[key]) for r in inmode]))

    def med(key, j):
        return float(np.median([r[key][j] for r in inmode if r[key] is not None]))

    pos = [x for p, _ in resid for x in p]
    neg = [x for _, n in resid for x in n]
    res = {"test": "T3.0 robust re-estimation on B0 captures (configs/support_hypotheses.json)",
           "measurements": len(rows), "in_mode_measurements": len(inmode), "clusters": len(clusters),
           "share_within_tolerance_in_mode": {"wls": share("wls"), "r3": share("r3"), "oracle": share("oracle")},
           "paired_difference_r3_minus_wls": {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                                              "improved": int((d > 0).sum()), "worsened": int((d < 0).sum())},
           "median_error_in_mode": {k: {"translation_m": med(k, 0), "yaw_deg": med(k, 1)} for k in ("wls", "r3", "oracle")},
           "r3_reflection_rejected": sum(r["r3"] is None for r in rows),
           "residual_auc_oracle_membership": None if not pos else 1 - auc(pos, neg),
           "oracle_correspondence_share": float(np.mean([r["n_oracle"] / r["n"] for r in inmode])),
           "passed": bool(lo > 0)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    with gzip.open(OUT.with_suffix(".rows.json.gz"), "wt") as fh:
        json.dump(rows, fh)
    print(json.dumps(res, indent=1))
