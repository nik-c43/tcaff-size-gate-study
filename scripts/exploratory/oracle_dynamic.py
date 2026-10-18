"""EXPLORATORY ORACLE: authors' detections minus detections near other robots/pedestrians (GT).

Not an algorithm (uses GT positions of movers); bounds the gain from removing dynamic objects.
Usage: python -I oracle_dynamic.py LAB_ROOT OUT_DIR RADIUS_M [RADIUS_M ...]
Writes OUT_DIR/B0_dynXXX/RR*.json (variant detections for scripts/segment_unit/run_variants.sh) and
OUT_DIR/oracle_dynamic_stats.json (kept in the repo as results/exploratory/oracle_dynamic/removal_stats.json).
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from robotdatapy.data import PoseData

LAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
RADII = [float(r) for r in sys.argv[3:]]
D = LAB / "data/tcaff_mot_data"
R = ["RR01", "RR04", "RR06", "RR08"]
params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
movers = {r: PoseData.from_bag(path=str(D / f"data/kimera_odom/{r}.bag"), topic=f"/{r}/world", time_tol=10., interp=True) for r in R}
for k in range(1, 7):
    movers[f"PED{k}"] = PoseData.from_bag(path=str(D / "data/pedestrian_gt.bag"), topic=f"/PED{k}/world", time_tol=10., interp=True)
stats = {}
for robot in R:
    frames = json.load(open(D / f"fastsam_data/{robot}.json"))
    T_BC = np.array(params[robot]["T_BC"]).reshape(4, 4)
    out = {r: [] for r in RADII}
    kept = {r: 0 for r in RADII}
    total = 0
    for fr in frames:
        t = fr["time"]
        dets = np.array(fr["detections"]).reshape(-1, 3)
        try:
            T_WB = movers[robot].T_WB(t)
            others = []
            for name, pd in movers.items():
                if name == robot:
                    continue
                try:
                    others.append(pd.T_WB(t)[:2, 3])
                except Exception:
                    pass
            others = np.array(others).reshape(-1, 2)
            W = (np.c_[dets, np.ones(len(dets))] @ (T_WB @ T_BC).T)[:, :2] if len(dets) else np.zeros((0, 2))
            dmin = np.min(np.linalg.norm(W[:, None] - others[None], axis=2), axis=1) if len(dets) and len(others) else np.full(len(dets), np.inf)
        except Exception:
            dmin = np.full(len(dets), np.inf)
        total += len(dets)
        for r in RADII:
            keep = [i for i in range(len(dets)) if dmin[i] >= r]
            kept[r] += len(keep)
            out[r].append({k: [fr[k][i] for i in keep] for k in ("detections", "pts_img", "covs_img", "widths", "heights")} | {"time": t})
    for r in RADII:
        d = OUT / f"B0_dyn{int(round(r * 100)):03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{robot}.json").write_text(json.dumps(out[r]))
        stats.setdefault(d.name, {})[robot] = {"detections": total, "kept": kept[r]}
(OUT / "oracle_dynamic_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
