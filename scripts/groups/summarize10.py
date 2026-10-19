"""Summarise test 10 full demo runs against the 5 B0 runs of check 4 (test10_protocol.json).
Usage: python summarize10.py RUNS_DIR OUT_JSON"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def run_metrics(path):
    traces = json.load(gzip.open(path))
    n = acc = pre = epsi = et = drop = gross = wrong_first = 0
    first_bad = []
    pair_acc = {}
    for tr in traces:
        seen, pa = False, 0
        for step, a, t, y in tr["frames"]:
            n += 1
            if not a:
                pre += not seen
                drop += seen
                continue
            ok = t <= 1.0 and y <= 5.0
            if not seen:
                if not ok:
                    wrong_first += 1
                    first_bad.append((tr["pair"], round(t, 2), round(y, 1)))
            seen = True
            if ok:
                acc += 1
                pa += 1
            elif t <= 1.0:
                epsi += 1
            else:
                et += 1
            gross += t > 2.0 or y > 15.0
        pair_acc[tr["pair"]] = pa / len(tr["frames"])
    return dict(acc=acc / n, u_pre=pre / n, e_psi=epsi / n, e_t=et / n, dropouts=drop / n, gross=gross / n,
                availability=1 - (pre + drop) / n, wrong_first=wrong_first, first_bad=first_bad, pair_acc=pair_acc)


def summarise(dirs):
    ms = [run_metrics(d / "error_traces.json.gz") for d in dirs]
    sums = [json.loads((d / "run_summary.json").read_text()) for d in dirs]
    out = {k: dict(mean=float(np.mean([m[k] for m in ms])), min=float(np.min([m[k] for m in ms])), max=float(np.max([m[k] for m in ms])))
           for k in ("acc", "u_pre", "e_psi", "e_t", "dropouts", "gross", "availability")}
    out["wrong_first_total"] = int(sum(m["wrong_first"] for m in ms))
    out["wrong_first_gross"] = int(sum(t > 2 or y > 15 for m in ms for _, t, y in m["first_bad"]))
    out["first_bad_examples"] = sorted({x for m in ms for x in m["first_bad"]})[:12]
    out["translation_m"] = float(np.mean([s["mean_of_pair_means_translation_m"] for s in sums]))
    out["yaw_deg"] = float(np.mean([s["mean_of_pair_means_yaw_deg"] for s in sums]))
    out["replay_mismatches"] = int(sum(s["replay_mismatches"] for s in sums))
    out["repeats"] = len(ms)
    out["pair_acc_mean"] = {p: float(np.mean([m["pair_acc"][p] for m in ms])) for p in ms[0]["pair_acc"]}
    return out


runs = sorted(Path(sys.argv[1]).glob("KBWA/rep*"))
res = {"B0": summarise(sorted((REPO / "results/segment_unit/unit_test/runs/B0").glob("rep*"))), "KBWA": summarise(runs)}
b, v = res["B0"], res["KBWA"]
rng = b["acc"]["max"] - b["acc"]["min"]
res["decision"] = dict(acc_gain_pp=100 * (v["acc"]["mean"] - b["acc"]["mean"]), b0_range_pp=100 * rng,
                       wrong_first=v["wrong_first_total"], b0_wrong_first=b["wrong_first_total"],
                       improves=bool(v["acc"]["mean"] - b["acc"]["mean"] > rng and v["wrong_first_total"] <= b["wrong_first_total"]))
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
f = lambda d: f"{100 * d['mean']:.2f} [{100 * d['min']:.2f}-{100 * d['max']:.2f}]"
for k in ("B0", "KBWA"):
    o = res[k]
    print(f"{k:5s} Acc {f(o['acc'])} Upre {f(o['u_pre'])} Epsi {f(o['e_psi'])} Et {f(o['e_t'])} gross {f(o['gross'])} "
          f"avail {f(o['availability'])} wrong1st {o['wrong_first_total']}/60 (gross {o['wrong_first_gross']}) "
          f"t {o['translation_m']:.3f} yaw {o['yaw_deg']:.2f} replay_mismatches {o['replay_mismatches']}")
print(json.dumps(res["decision"], indent=1))
for p in sorted(b["pair_acc_mean"]):
    print(p, f"{100 * b['pair_acc_mean'][p]:6.1f} {100 * v['pair_acc_mean'][p]:6.1f}")
