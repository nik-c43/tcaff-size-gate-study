"""Summarise test 2 (test2.py output). Usage: python summarize2.py RUNS_DIR OUT_JSON"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB = Path(os.environ["LAB_ROOT"])
runs = Path(sys.argv[1])
rows = [r for p in sorted(runs.glob("*.json")) for r in json.loads(p.read_text())]

# B0 first acceptance per directed pair in the frozen release run (for pre-acceptance coverage)
first_acc = {}
for p in sorted((LAB / "data/tcaff_baseline_release").glob("RR*_RR*.json")):
    fr = json.loads(p.read_text())["frames"]
    first_acc[p.stem] = next((f["step"] for f in fr if f["baseline_estimate"] is not None), 10 ** 9)

correct = lambda r: r["t"] <= 1.0 and r["yaw"] <= 5.0
in_mode = lambda r: r["t"] <= 2.0 and r["yaw"] <= 15.0
q = lambda v, ps=(0.5, 0.9): np.quantile(v, ps).round(3).tolist() if len(v) else None
res = {}
for cfg in sorted({r["config"] for r in rows}):
    steps = [r for r in rows if r["config"] == cfg and r["kind"] == "step"]
    meas = [r for r in rows if r["config"] == cfg and r["kind"] == "meas"]
    by_step = defaultdict(list)
    for m in meas:
        by_step[(m["pair"], m["step"])].append(m)
    pre = [s for s in steps if s["step"] < first_acc[s["pair"]]]
    wrong = [m for m in meas if not in_mode(m)]
    good = [m for m in meas if in_mode(m)]
    out = dict(
        steps=len(steps),
        steps_with_measurement=float(np.mean([s["n_meas"] > 0 for s in steps])),
        steps_with_correct=float(np.mean([any(correct(m) for m in by_step[(s["pair"], s["step"])]) for s in steps])),
        steps_with_wrong_mode=float(np.mean([any(not in_mode(m) for m in by_step[(s["pair"], s["step"])]) for s in steps])),
        pre_acceptance_steps=len(pre),
        pre_steps_with_correct=float(np.mean([any(correct(m) for m in by_step[(s["pair"], s["step"])]) for s in pre])) if pre else None,
        measurements=len(meas), correct=sum(map(correct, meas)), in_mode=len(good), wrong_mode=len(wrong),
        in_mode_within_threshold=float(np.mean([correct(m) for m in good])) if good else None,
        in_mode_yaw_q50_q90=q([m["yaw"] for m in good]), in_mode_t_q50_q90=q([m["t"] for m in good]),
        wrong_mode_by_pair={p: sum(m["pair"] == p for m in wrong) for p in sorted({m["pair"] for m in wrong})},
        wrong_with_cross_group=float(np.mean([m["cross"] > 0 for m in wrong])) if wrong else None,
        wrong_cross_share_of_in_group=float(np.sum([m["cross"] for m in wrong]) / max(1, np.sum([m["in_groups"] for m in wrong]))) if wrong else None,
        wrong_without_group_objects=float(np.mean([m["in_groups"] == 0 for m in wrong])) if wrong else None,
        wrong_in_group_share_of_corr=float(np.sum([m["in_groups"] for m in wrong]) / max(1, np.sum([m["n"] for m in wrong]))) if wrong else None,
        good_with_cross_group=float(np.mean([m["cross"] > 0 for m in good])) if good else None,
        good_in_group_share_of_corr=float(np.sum([m["in_groups"] for m in good]) / max(1, np.sum([m["n"] for m in good]))) if good else None,
    )
    # precision of in-mode measurements vs clique extent and number of GT groups spanned
    for name, key, bins in (("extent_m", "extent", [0, 1.5, 3, 5, 100]), ("groups_spanned", "groups_a", [0, 1, 2, 3, 10])):
        tab = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = [m for m in good if lo <= m[key] < hi]
            if sel:
                tab[f"[{lo},{hi})"] = dict(n=len(sel), within_threshold=round(float(np.mean([correct(m) for m in sel])), 3),
                                           yaw_q50=round(float(np.median([m["yaw"] for m in sel])), 2),
                                           t_q50=round(float(np.median([m["t"] for m in sel])), 3))
        out[f"in_mode_by_{name}"] = tab
    res[cfg] = out
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
for cfg, o in res.items():
    print(cfg, {k: o[k] for k in ("steps_with_correct", "pre_steps_with_correct", "steps_with_wrong_mode", "measurements",
                                   "correct", "wrong_mode", "in_mode_within_threshold", "wrong_with_cross_group",
                                   "wrong_without_group_objects")})
