"""Where do the 798 labelled pairs lie, and do robot/person notes agree with GT? (report §5)

Each side's last observation (frame and detection index in the authors' FastSAM JSON) is placed in
the mocap world frame with the robot's GT pose. Place classes as in landmarks.py. "Near mover" =
within 0.3 m XY of another robot or a pedestrian (GT) at the observation time.

Usage: python -I annotation_places.py WORKSPACE OUT_JSON
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from robotdatapy.data import PoseData

LAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
REPO = Path(__file__).resolve().parents[2]
D = LAB / "data/tcaff_mot_data"
params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
boxes = np.loadtxt(D / "objects.txt", delimiter=",")
tasks = json.load(open(REPO / "annotations/task_metadata.json"))["tasks"]
tasks = tasks if isinstance(tasks, list) else list(tasks.values())
rows = {r["task_id"]: r for r in csv.DictReader(open(REPO / "annotations/labels.csv"))}
R = params["robots"]
gt = {r: PoseData.from_bag(path=str(D / f"data/kimera_odom/{r}.bag"), topic=f"/{r}/world", time_tol=10., interp=True) for r in R}
movers = dict(gt)
for k in range(1, 7):
    movers[f"PED{k}"] = PoseData.from_bag(path=str(D / "data/pedestrian_gt.bag"), topic=f"/PED{k}/world", time_tol=10., interp=True)
dets = {r: json.load(open(D / f"fastsam_data/{r}.json")) for r in R}


def world(robot, fi, di):
    c = np.array(dets[robot][fi]["detections"][di])
    t = dets[robot][fi]["time"]
    return (gt[robot].T_WB(t) @ np.array(params[robot]["T_BC"]).reshape(4, 4) @ np.r_[c, 1.0])[:3], t


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


def nearest_mover(p, t, own):
    best = np.inf
    for name, pd in movers.items():
        if name != own:
            try:
                best = min(best, float(np.linalg.norm(pd.T_WB(t)[:2, 3] - p[:2])))
            except Exception:
                pass
    return best


def note(n):
    n = n.lower()
    return "robot" if "робот" in n else "person" if "человек" in n else "none"


sides, by_answer = Counter(), defaultdict(Counter)
notes = defaultdict(Counter)
noted_dist = []
for t in tasks:
    r = rows[t["task_id"]]
    places, dmin = [], []
    for s in t["sides"]:
        v = s["views"][0]
        p, tt = world(s["robot"], v["frame_index"], v["detection_index"])
        places.append(place(p))
        dmin.append(nearest_mover(p, tt, s["robot"]))
    for pl in places:
        sides[pl] += 1
    by_answer[r["label"]][places[0] if places[0] == places[1] else "mixed"] += 1
    kind = note(r["notes"])
    near = min(dmin) < 0.3
    notes[kind][f"near_mover={near}"] += 1
    if kind == "robot":
        noted_dist.append(min(dmin))
    elif kind == "none" and near:
        notes["none_near_mover_answers"][r["label"]] += 1
res = {"pairs": len(tasks), "sides_by_place": dict(sides), "pairs_by_answer_and_place": {k: dict(v) for k, v in by_answer.items()},
       "notes_vs_gt": {k: dict(v) for k, v in notes.items()},
       "robot_noted_min_distance_m_q25_q50_q75": np.quantile(noted_dist, [.25, .5, .75]).round(3).tolist() if noted_dist else None}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
