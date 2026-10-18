"""Summarize the exploratory dynamic-object oracle against the B0 repeats of the segment-unit test.

Usage: python3 -I oracle_summary.py   (from any directory; needs numpy)
Writes results/exploratory/oracle_dynamic/summary.json.
"""
import gzip
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/exploratory/oracle_dynamic"
DIRS = {"B0": REPO / "results/segment_unit/unit_test/runs/B0",
        "B0_dyn030": OUT / "runs/B0_dyn030", "B0_dyn060": OUT / "runs/B0_dyn060"}


def metrics(rep):
    s = json.loads((rep / "run_summary.json").read_text())
    assert s["replay_mismatches"] == 0, rep
    n = yaw10 = acc = gross = 0
    for tr in json.load(gzip.open(rep / "error_traces.json.gz")):
        for _, a, t, y in tr["frames"]:
            n += 1
            acc += bool(a and t <= 1.0 and y <= 5.0)
            yaw10 += bool(a and y > 10.0)
            gross += bool(a and (t > 2.0 or y > 15.0))
    return dict(acc=acc / n, availability=s["availability"], translation_m=s["mean_of_pair_means_translation_m"],
                yaw_deg=s["mean_of_pair_means_yaw_deg"], share_updates_yaw_gt10=yaw10 / n, share_updates_gross=gross / n)


stats = json.loads((OUT / "removal_stats.json").read_text())
res = {"note": "Exploratory, not pre-registered; removes detections within RADIUS of another robot or pedestrian using GT positions (an oracle, not an algorithm)."}
for v, d in DIRS.items():
    rows = [metrics(r) for r in sorted(d.glob("rep*"))]
    res[v] = {"repeats": len(rows), "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
              "range": {k: float(np.ptp([r[k] for r in rows])) for k in rows[0]}}
    if v in stats:
        det = sum(x["detections"] for x in stats[v].values())
        res[v]["removed_detection_share"] = 1 - sum(x["kept"] for x in stats[v].values()) / det
(OUT / "summary.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
