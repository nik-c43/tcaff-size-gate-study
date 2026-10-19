"""Summarise test 7 together with test-5 references. Usage: python summarize7.py OUT_JSON RUNS_DIR [RUNS_DIR ...]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

runs = defaultdict(list)
for p in sorted(q for d in sys.argv[2:] for q in Path(d).glob("*.json")):
    rows = json.loads(p.read_text())
    runs[(rows[0]["config"], rows[0]["rep"])].append(rows)
ok = lambda r: r["accepted"] and r["t"] <= 1.0 and r["yaw"] <= 5.0


def metrics(traces):
    n = acc = pre = epsi = et = drop = gross = 0
    wrong_first, first_steps, pair_t, pair_y, pre_meas, pre_meas_ok = 0, [], [], [], 0, 0
    per_pair = {}
    for tr in traces:
        seen, ts, ys, pacc = False, [], [], 0
        for r in tr:
            n += 1
            if not r["accepted"]:
                if not seen:
                    pre += 1
                    pre_meas += 1
                    pre_meas_ok += r["n_meas_correct"] > 0
                else:
                    drop += 1
                continue
            if not seen:
                first_steps.append(r["step"])
                wrong_first += not ok(r)
            seen = True
            ts.append(r["t"]); ys.append(r["yaw"])
            if ok(r):
                acc += 1; pacc += 1
            elif r["t"] <= 1.0:
                epsi += 1
            else:
                et += 1
            gross += r["t"] > 2.0 or r["yaw"] > 15.0
        if ts:
            pair_t.append(np.mean(ts)); pair_y.append(np.mean(ys))
        per_pair[tr[0]["pair"]] = pacc / len(tr)
    return dict(acc=acc / n, u_pre=pre / n, e_psi=epsi / n, e_t=et / n, dropouts=drop / n, gross=gross / n,
                availability=1 - (pre + drop) / n, wrong_first=wrong_first, first_acceptances=len(first_steps),
                first_step_median=float(np.median(first_steps)) if first_steps else None,
                pre_steps_with_correct_measurement=pre_meas_ok / max(1, pre_meas),
                translation_m=float(np.mean(pair_t)), yaw_deg=float(np.mean(pair_y)), per_pair=per_pair)


res = {}
for cfg in ["B0", "B0+KO", "B0+KB", "B0+KB4", "B0+KB3RH", "B0+KB4RH", "B0+KOC", "B0+KBT", "B0+KBWA", "B0+KBWB"]:
    reps = sorted(r for c, r in runs if c == cfg)
    if not reps:
        continue
    ms = [metrics(runs[(cfg, r)]) for r in reps]
    keys = [k for k in ms[0] if k != "per_pair"]
    out = {k: dict(mean=float(np.mean([m[k] for m in ms])), min=float(np.min([m[k] for m in ms])),
                   max=float(np.max([m[k] for m in ms]))) for k in keys if ms[0][k] is not None}
    out["repeats"] = len(ms)
    out["traces_per_repeat"] = [len(runs[(cfg, r)]) for r in reps]
    out["per_pair_acc_mean"] = {p: float(np.mean([m["per_pair"][p] for m in ms])) for p in ms[0]["per_pair"]}
    res[cfg] = out
b = res["B0"]
b_range = b["acc"]["max"] - b["acc"]["min"]
res["decision"] = {}
for cfg in ["B0+KO", "B0+KB", "B0+KB4", "B0+KB3RH", "B0+KB4RH", "B0+KOC", "B0+KBT", "B0+KBWA", "B0+KBWB"]:
    if cfg in res:
        o = res[cfg]
        res["decision"][cfg] = dict(acc_gain_pp=100 * (o["acc"]["mean"] - b["acc"]["mean"]), b0_range_pp=100 * b_range,
                                    wrong_first_mean=o["wrong_first"]["mean"], b0_wrong_first_mean=b["wrong_first"]["mean"],
                                    improves=bool(o["acc"]["mean"] - b["acc"]["mean"] > b_range and o["wrong_first"]["mean"] <= b["wrong_first"]["mean"]))
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
fmt = lambda d: f"{100 * d['mean']:.2f} [{100 * d['min']:.2f}-{100 * d['max']:.2f}]"
for cfg, o in res.items():
    if cfg == "decision":
        continue
    print(f"{cfg:8s} Acc {fmt(o['acc'])} Upre {fmt(o['u_pre'])} Epsi {fmt(o['e_psi'])} Et {fmt(o['e_t'])} gross {fmt(o['gross'])} "
          f"avail {fmt(o['availability'])} wrong1st {o['wrong_first']['mean']:.1f}/12 first {o['first_step_median']['mean']:.1f} "
          f"pre-corr-meas {fmt(o['pre_steps_with_correct_measurement'])} t {o['translation_m']['mean']:.3f} yaw {o['yaw_deg']['mean']:.2f}")
print(json.dumps(res["decision"], indent=1))
