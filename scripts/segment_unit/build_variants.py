"""Build detection files for variants R0, U1, U2, U3 (configs/unit_hypothesis.json) and mechanism metrics.

Output per variant: <OUT>/<V>/<ROBOT>.json in the authors' FastSAM JSON format.

Usage: python -I build_variants.py FULL_DIR OUT_DIR
"""
import json
import sys
from collections import Counter
from pathlib import Path

FULL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ROBOTS = ["RR01", "RR04", "RR06", "RR08"]
CONTAIN_SHARE, LAYER_SHARE = 0.8, 0.10  # containment computed in segment_all.py at 0.8


def multi_layer(m):
    return m["frac_behind"] >= LAYER_SHARE or m["frac_front"] >= LAYER_SHARE


def select(masks):
    r0 = [m for m in masks if "c_cam" in m and not m["border"]]
    ids = {m["i"] for m in r0}
    container = {m["i"] for m in r0 if any(j in ids for j in m["contains"])}
    layered = {m["i"] for m in r0 if multi_layer(m)}
    return {"R0": r0,
            "U1": [m for m in r0 if m["i"] not in container],
            "U2": [m for m in r0 if m["i"] not in layered],
            "U3": [m for m in r0 if m["i"] not in container and m["i"] not in layered]}


mech = {v: Counter() for v in ["R0", "U1", "U2", "U3"]}
for R in ROBOTS:
    files = {v: [] for v in mech}
    for line in open(FULL / R / "masks.jsonl"):
        fr = json.loads(line)
        chosen = select(fr["masks"] or [])
        for v, ms in chosen.items():
            files[v].append({"detections": [m["c_cam"] for m in ms], "pts_img": [m["pts_img"] for m in ms],
                             "covs_img": [m["cov_img"] for m in ms], "widths": [m["w"] for m in ms],
                             "heights": [m["h"] for m in ms], "time": fr["time"]})
            c = mech[v]
            c["frames"] += 1
            c["detections"] += len(ms)
            box = [m for m in ms if m.get("share_box", 0) >= 0.6]
            c["box_related"] += len(box)
            c["box_multi"] += sum(len(m["boxes"]) >= 2 for m in box)
            per_box = Counter(b for m in ms if m.get("share_box", 0) >= 0.2 for b in m.get("boxes", []))
            c["box_observations"] += len(per_box)
            c["box_observations_dup"] += sum(k >= 2 for k in per_box.values())
    for v, rows in files.items():
        (OUT / v).mkdir(parents=True, exist_ok=True)
        (OUT / v / f"{R}.json").write_text(json.dumps(rows))
summary = {v: {"detections_per_frame": c["detections"] / c["frames"],
               "box_related_detections": c["box_related"],
               "multi_box_share_of_box_related": c["box_multi"] / max(1, c["box_related"]),
               "box_observations_in_two_or_more_detections": c["box_observations_dup"] / max(1, c["box_observations"]),
               "frames": c["frames"]} for v, c in mech.items()}
(OUT / "mechanism.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
