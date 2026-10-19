"""Collect every robot's TCAFF map (odom frame) at every step of the frozen B0 release run,
with on-board context (robot odom pose) and GT-only evaluation fields (world position, place class,
GT box and group). Output: one npz with a table of all robot-step map objects.
Usage: LAB_ROOT=... python -I collect_maps.py MAPS_NPZ"""
import json, os, sys
from pathlib import Path
import numpy as np, yaml
from scipy.cluster.hierarchy import linkage, fcluster
LAB = Path(os.environ["LAB_ROOT"])
OUT = Path(sys.argv[1])
os.environ["TCAFF_MOT_DATASET"] = str(LAB / "data/tcaff_mot_data")
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff")]
import tcaff_baseline_audit as audit
from robotdatapy.data import PoseData
D = LAB / "data/tcaff_mot_data"
params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
poses = audit.load_poses(params)
boxes = np.loadtxt(D / "objects.txt", delimiter=",")
gt_group = fcluster(linkage(boxes, "single"), 0.7, "distance") - 1   # GT groups, evaluation only
movers = {r: poses[(r, "pose_gt")] for r in params["robots"]}
for k in range(1, 7):
    movers[f"PED{k}"] = PoseData.from_bag(path=str(D / "data/pedestrian_gt.bag"), topic=f"/PED{k}/world", time_tol=10., interp=True)
robots = params["robots"]
rows = []
for r in robots:
    other = next(o for o in robots if o != r)
    d = json.load(open(LAB / f"data/tcaff_baseline_release/{r}_{other}.json"))
    for fr in d["frames"]:
        M = np.asarray(fr["map_a"], float)
        if not len(M):
            continue
        t = fr["time"]
        try:
            T_wb = poses[(r, "pose_gt")].T_WB(t)
            T_ob = poses[(r, "pose_estimate")].T_WB(t)
        except Exception:
            continue
        T = T_wb @ np.linalg.inv(T_ob)
        W = (np.c_[M[:, :3], np.ones(len(M))] @ T.T)[:, :3]
        dist_box = np.linalg.norm(W[:, None, :2] - boxes[None], axis=2)
        nb = dist_box.argmin(1); db = dist_box.min(1)
        mov = []
        for name, pd in movers.items():
            if name == r:
                continue
            try:
                mov.append(pd.T_WB(t)[:2, 3])
            except Exception:
                pass
        dmov = np.linalg.norm(W[:, None, :2] - np.array(mov)[None], axis=2).min(1) if mov else np.full(len(W), np.inf)
        for i in range(len(M)):
            rows.append([robots.index(r), fr["step"], t, *M[i, :6], *T_ob[:3, 3], *W[i], db[i], nb[i],
                         gt_group[nb[i]] if (db[i] < 0.35 and W[i, 2] < 0.9 and W[i, 2] >= 0.08) else -1, dmov[i]])
cols = "robot step time x y z w h age rx ry rz wx wy wz dbox nbox ggroup dmover".split()
np.savez_compressed(OUT, data=np.array(rows), cols=np.array(cols), robots=np.array(robots), boxes=boxes, gt_group=gt_group)
print(len(rows), "rows")
