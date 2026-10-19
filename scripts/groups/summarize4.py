"""Summarise test 4 and apply the decision rule of test4_protocol.json. Usage: python summarize4.py RUNS_DIR OUT_JSON"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB = Path(os.environ["LAB_ROOT"])
rows = [r for p in sorted(Path(sys.argv[1]).glob("*.json")) for r in json.loads(p.read_text())]
first_acc = {}
for p in sorted((LAB / "data/tcaff_baseline_release").glob("RR*_RR*.json")):
    fr = json.loads(p.read_text())["frames"]
    first_acc[p.stem] = next((f["step"] for f in fr if f["baseline_estimate"] is not None), 10 ** 9)
correct = lambda r: r["t"] <= 1.0 and r["yaw"] <= 5.0
in_mode = lambda r: r["t"] <= 2.0 and r["yaw"] <= 15.0
res = {}
for cfg in ["B0", "G2", "B0+KO", "B0+KB"]:
    steps = [r for r in rows if r["config"] == cfg and r["kind"] == "step"]
    meas = [r for r in rows if r["config"] == cfg and r["kind"] == "meas"]
    by = defaultdict(list)
    for m in meas:
        by[(m["pair"], m["step"])].append(m)
    has_correct = lambda s: any(correct(m) for m in by[(s["pair"], s["step"])])
    pre = [s for s in steps if s["step"] < first_acc[s["pair"]]]
    wrong = [m for m in meas if not in_mode(m)]
    good = [m for m in meas if in_mode(m)]
    withc = [m for m in meas if m["n_cluster"] > 0]
    res[cfg] = dict(
        steps=len(steps), pre_acceptance_steps=len(pre),
        pre_steps_with_correct=float(np.mean([has_correct(s) for s in pre])),
        steps_with_correct=float(np.mean([has_correct(s) for s in steps])),
        steps_with_wrong_mode=float(np.mean([any(not in_mode(m) for m in by[(s["pair"], s["step"])]) for s in steps])),
        measurements=len(meas), correct=sum(map(correct, meas)), wrong_mode=len(wrong),
        in_mode_within_threshold=float(np.mean([correct(m) for m in good])) if good else None,
        clusters_per_map_mean=float(np.mean([s["k1"] for s in steps])),
        meas_with_cluster_nodes=float(len(withc) / max(1, len(meas))),
        cluster_nodes_per_meas_with_clusters=float(np.mean([m["n_cluster"] for m in withc])) if withc else None,
        wrong_with_cluster_nodes=float(np.mean([m["n_cluster"] > 0 for m in wrong])) if wrong else None,
        wrong_with_wrong_identity_cluster=float(np.mean([m["cluster_wrong"] > 0 for m in wrong])) if wrong else None,
        correct_with_cluster_nodes=float(np.mean([m["n_cluster"] > 0 for m in meas if correct(m)])) if meas else None,
        correct_with_wrong_identity_cluster=float(np.mean([m["cluster_wrong"] > 0 for m in meas if correct(m)])) if meas else None,
        wrong_mode_by_pair={p: sum(m["pair"] == p for m in wrong) for p in sorted({m["pair"] for m in wrong})})
ko, g2 = res["B0+KO"], res["G2"]
res["decision"] = dict(
    coverage_at_least_G2=ko["pre_steps_with_correct"] >= g2["pre_steps_with_correct"],
    wrong_at_most_75pct_of_G2=ko["wrong_mode"] <= 0.75 * g2["wrong_mode"],
    verdict="promising" if (ko["pre_steps_with_correct"] >= g2["pre_steps_with_correct"] and ko["wrong_mode"] <= 0.75 * g2["wrong_mode"])
    else "not better than relaxing the gate")
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
for cfg in ["B0", "G2", "B0+KO", "B0+KB"]:
    o = res[cfg]
    print(cfg, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "wrong_mode_by_pair"})
print(res["decision"])
