"""Aggregate repeats per variant and apply the decision rule of configs/unit_hypothesis.json.

Usage: python3 aggregate.py RUNS_DIR MECHANISM_JSON OUT_JSON
"""
import gzip
import json
import statistics
import sys
from pathlib import Path

RUNS, MECH, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
METRICS = ["correct_share_all_updates", "mean_of_pair_means_translation_m", "mean_of_pair_means_yaw_deg",
           "availability", "pooled_mean_translation_m"]
mech = json.loads(MECH.read_text())
res = {"variants": {}, "mechanism": mech}
for v in ["R0", "U1", "U2", "U3", "B0"]:  # B0: authors' detections, reference only, not in the decision
    if v == "B0" and not (RUNS / v).exists():
        continue
    runs = [json.loads(p.read_text()) for p in sorted((RUNS / v).glob("rep*/run_summary.json"))]
    assert runs and all(r["replay_mismatches"] == 0 for r in runs), v
    agg = {"repeats": len(runs)}
    for m in METRICS:
        xs = [r[m] for r in runs]
        agg[m] = {"mean": statistics.mean(xs), "min": min(xs), "max": max(xs), "values": xs}
    firsts = [t for r in runs for t in r["first_acceptance_s"].values() if t is not None]
    agg["first_acceptance_s_median"] = statistics.median(firsts) if firsts else None
    agg["pairs_without_acceptance"] = sum(t is None for r in runs for t in r["first_acceptance_s"].values())
    res["variants"][v] = agg

# Per directed pair, averaged over repeats (from error_traces: [step, accepted, translation_m, yaw_deg]).
res["per_pair"] = {}
for v in res["variants"]:
    by = {}
    for path in sorted((RUNS / v).glob("rep*/error_traces.json.gz")):
        for t in json.load(gzip.open(path, "rt")):
            by.setdefault(t["pair"], []).append(t["frames"])
    out = {}
    for pair, reps in sorted(by.items()):
        corr = [sum(1 for f in fr if f[2] is not None and f[2] <= 1.0 and f[3] <= 5.0) / len(fr) for fr in reps]
        acc = [sum(f[1] for f in fr) / len(fr) for fr in reps]
        tr = [statistics.mean(f[2] for f in fr if f[2] is not None) for fr in reps if any(f[2] is not None for f in fr)]
        yw = [statistics.mean(f[3] for f in fr if f[3] is not None) for fr in reps if any(f[3] is not None for f in fr)]
        out[pair] = {"correct_share_mean": statistics.mean(corr), "correct_share_min": min(corr), "correct_share_max": max(corr),
                     "availability_mean": statistics.mean(acc),
                     "translation_m_mean": statistics.mean(tr) if tr else None, "yaw_deg_mean": statistics.mean(yw) if yw else None}
    res["per_pair"][v] = out

r0, u3 = res["variants"]["R0"], res["variants"]["U3"]
d_correct = u3["correct_share_all_updates"]["mean"] - r0["correct_share_all_updates"]["mean"]
d_trans = r0["mean_of_pair_means_translation_m"]["mean"] - u3["mean_of_pair_means_translation_m"]["mean"]
range_correct = r0["correct_share_all_updates"]["max"] - r0["correct_share_all_updates"]["min"]
range_trans = r0["mean_of_pair_means_translation_m"]["max"] - r0["mean_of_pair_means_translation_m"]["min"]
d_avail_pp = 100 * (r0["availability"]["mean"] - u3["availability"]["mean"])
d_multi = mech["R0"]["multi_box_share_of_box_related"] - mech["U3"]["multi_box_share_of_box_related"]
checks = {"correct_share_higher_beyond_R0_range": d_correct > 0 and d_correct > range_correct,
          "translation_lower_beyond_R0_range": d_trans > 0 and d_trans > range_trans,
          "availability_drop_at_most_5pp": d_avail_pp <= 5.0,
          "multi_box_share_decreases": d_multi > 0}
res["decision"] = {"U3_minus_R0_correct_share": d_correct, "R0_range_correct_share": range_correct,
                   "R0_minus_U3_translation_m": d_trans, "R0_range_translation_m": range_trans,
                   "availability_drop_pp": d_avail_pp, "multi_box_share_drop": d_multi,
                   "checks": checks, "supported": all(checks.values())}
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({"decision": res["decision"],
                  "summary": {v: {m: round(a[m]["mean"], 4) for m in METRICS} for v, a in res["variants"].items()}}, indent=1))
