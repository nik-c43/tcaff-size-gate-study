"""What do TCAFF map objects and CLIPPER inlier correspondences sit on? (docs/2_diagnostics.md; check 4 in docs/3_hypotheses.md)

Replays frame alignment (release TCAFFManager) on the maps captured during a run, for every STEP-th
step, and keeps the solution closest to the run's accepted estimate. Map objects are placed in the
mocap world frame: T_world_odom = T_world_body(GT) @ inv(T_odom_body(estimate)).

Place classes (geometric, not identity labels):
  floor        z < 0.08 m
  gt_obstacle  within 0.35 m XY of an objects.txt point and z < 0.9 m (dataset README: static boxes)
  room_edge    |x| > 4.5 m or |y| > 4.5 m (walls and things along walls)
  high         z > 1.2 m
  other        the rest (inside the room; in the labelled pool mostly robots)
Dynamic: within 0.3 m XY of another robot or a pedestrian (GT) at the step time.
CLIPPER is stochastic, so the replayed cliques approximate those of the run.

Usage: python -I landmarks.py WORKSPACE CAPTURE_DIR OUT_JSON [STEP] [FAILING_PAIRS_COMMA_SEPARATED]
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

LAB, CAP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 10
FAIL = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set()
os.environ["TCAFF_MOT_DATASET"] = str(LAB / "data/tcaff_mot_data")
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff")]
import tcaff_baseline_audit as audit  # noqa: E402
from robotdatapy.data import PoseData  # noqa: E402
from tcaff.realign.object_map import ObjectMap  # noqa: E402
from tcaff.tcaff.tcaff_manager import TCAFFManager  # noqa: E402
from tcaff.utils.transform import transform_2_xypsi  # noqa: E402

D = LAB / "data/tcaff_mot_data"
params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
poses = audit.load_poses(params)
boxes = np.loadtxt(D / "objects.txt", delimiter=",")
movers = {r: poses[(r, "pose_gt")] for r in params["robots"]}
for k in range(1, 7):
    movers[f"PED{k}"] = PoseData.from_bag(path=str(D / "data/pedestrian_gt.bag"), topic=f"/PED{k}/world", time_tol=10., interp=True)
kwargs = {k: v for k, v in params["tcaff"].items() if k != "ts"}


def to_world(robot, t, M):
    T = poses[(robot, "pose_gt")].T_WB(t) @ np.linalg.inv(poses[(robot, "pose_estimate")].T_WB(t))
    return (np.c_[M[:, :3], np.ones(len(M))] @ T.T)[:, :3]


def place(p):
    d = np.min(np.linalg.norm(boxes - p[:2], axis=1))
    if p[2] < 0.08:
        return "floor"
    if d < 0.35 and p[2] < 0.9:
        return "gt_obstacle"
    if abs(p[0]) > 4.5 or abs(p[1]) > 4.5:
        return "room_edge"
    if p[2] > 1.2:
        return "high"
    return "other"


def dynamic(p, t, own):
    for name, pd in movers.items():
        if name == own:
            continue
        try:
            if np.linalg.norm(pd.T_WB(t)[:2, 3] - p[:2]) < 0.3:
                return True
        except Exception:
            pass
    return False


groups = {"all": Counter(), "failing_pairs": Counter(), "other_pairs": Counter()}
map_objects = Counter()
for path in sorted(CAP.glob("*.json")):
    data = json.loads(path.read_text())
    a, b = data["pair"].split("_")
    mgr = TCAFFManager(**kwargs)
    for fr in data["frames"][::STEP]:
        if not fr["map_a"] or not fr["map_b"]:
            continue
        A, B = np.array(fr["map_a"]), np.array(fr["map_b"])
        WA, WB = to_world(a, fr["time"], A), to_world(b, fr["time"], B)
        for p in WA:
            map_objects[place(p)] += 1
        mgr.get_frame_align_measurements(ObjectMap(centroids=A[:, :3], widths=A[:, 3], heights=A[:, 4], ages=A[:, 5]),
                                         ObjectMap(centroids=B[:, :3], widths=B[:, 3], heights=B[:, 4], ages=B[:, 5]))
        if not mgr.latest_fa_solutions or fr["baseline_estimate"] is None:
            continue
        ref = np.array(fr["baseline_estimate"])

        def gap(s):
            x = np.array(transform_2_xypsi(s.transform))
            return np.hypot(*(x[:2] - ref[:2])) + abs(np.arctan2(np.sin(x[2] - ref[2]), np.cos(x[2] - ref[2])))
        sol = min(mgr.latest_fa_solutions, key=gap)
        for i, j in np.asarray(sol.associated_objs).reshape(-1, 2).astype(int):
            pa, pb = place(WA[i]), place(WB[j])
            kind = pa if pa == pb else "mixed"
            right = bool(np.linalg.norm(WA[i, :2] - WB[j, :2]) < 0.5)
            dyn = dynamic(WA[i], fr["time"], a) or dynamic(WB[j], fr["time"], b)
            for g in ("all", "failing_pairs" if data["pair"] in FAIL else "other_pairs"):
                c = groups[g]
                c["inliers"] += 1
                c[f"kind:{kind}"] += 1
                c["same_place" if right else "wrong_place"] += 1
                if not right:
                    c[f"wrong_kind:{kind}"] += 1
                    c["wrong_dynamic"] += dyn
                else:
                    c["right_dynamic"] += dyn
res = {"capture": CAP.parent.name + "/" + CAP.name, "step_stride": STEP, "failing_pairs": sorted(FAIL),
       "map_objects_robot_a": dict(map_objects),
       "inliers": {g: dict(c) for g, c in groups.items() if c}}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
