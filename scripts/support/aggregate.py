"""Aggregate the H2/H3 protocol and apply its decision rules (configs/support_hypotheses.json).

Usage: python -I aggregate.py RESULTS_DIR
RESULTS_DIR holds runs/{G2,S2,R3}/rep*/{run_summary.json,error_traces.json.gz,timing.json},
cold_start/{B0,G2,S2,R3}.json, coverage.json, robust_offline.json, size_auc.json.
B0 full-recording runs are read from results/segment_unit/unit_test/runs/B0.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1])
RUN_DIRS = {"B0": REPO / "results/segment_unit/unit_test/runs/B0", **{v: RES / "runs" / v for v in ("G2", "S2", "R3")}}


def decompose(path):
    n = acc = pre = e_psi = e_t = big_yaw = gross = 0
    for tr in json.load(gzip.open(path)):
        seen = False
        for _, accepted, t, yaw in tr["frames"]:
            n += 1
            if not accepted:
                pre += not seen
                continue
            seen = True
            if t <= 1.0 and yaw <= 5.0:
                acc += 1
            elif t <= 1.0:
                e_psi += 1
            else:
                e_t += 1
            big_yaw += yaw > 10.0
            gross += t > 2.0 or yaw > 15.0
    accepted_n = n - pre
    return dict(acc=acc / n, u_pre=pre / n, e_psi=e_psi / n, e_t=e_t / n, e_sum=(e_psi + e_t) / n,
                u_post=1 - acc / n - pre / n - (e_psi + e_t) / n, gross=gross / n,
                share_yaw_gt10_of_accepted=big_yaw / max(1, accepted_n))


def error_profile(paths):
    """Accepted-error profile over all repeats: maxima, quantiles of near-threshold errors, wrong first acceptances."""
    t_all, y_all, y_angle, t_trans, first_wrong, starts = [], [], [], [], 0, 0
    for path in paths:
        for tr in json.load(gzip.open(path)):
            first = True
            for _, a, t, y in tr["frames"]:
                if not a:
                    continue
                t_all.append(t)
                y_all.append(y)
                if t <= 1.0 and y > 5.0:
                    y_angle.append(y)
                if t > 1.0:
                    t_trans.append(t)
                if first:
                    starts += 1
                    first_wrong += not (t <= 1.0 and y <= 5.0)
                    first = False
    q = lambda v, p: np.quantile(v, p).round(3).tolist() if v else None
    return dict(max_translation_m=float(max(t_all)), max_yaw_deg=float(max(y_all)),
                angle_error_yaw_q50_q90=q(y_angle, [0.5, 0.9]), translation_error_t_q25_q50_q75=q(t_trans, [0.25, 0.5, 0.75]),
                first_acceptance_wrong=first_wrong, first_acceptances=starts)


def per_pair(path):
    """Per directed pair: Acc, U_pre, E_t, share of updates with |yaw| > 10 deg and the longest run of them."""
    out = {}
    for tr in json.load(gzip.open(path)):
        f = tr["frames"]
        seen, pre, run, longest = False, 0, 0, 0
        for _, a, t, y in f:
            seen = seen or a
            pre += not seen
            run = run + 1 if a and y > 10.0 else 0
            longest = max(longest, run)
        out[tr["pair"]] = dict(acc=sum(1 for _, a, t, y in f if a and t <= 1.0 and y <= 5.0) / len(f), u_pre=pre / len(f),
                               e_t=sum(1 for _, a, t, y in f if a and t > 1.0) / len(f),
                               yaw_gt10=sum(1 for _, a, t, y in f if a and y > 10.0) / len(f), longest_yaw_gt10_run=longest)
    return out


def oracle_detail(rows_path):
    rows = json.load(gzip.open(rows_path))
    inmode = [r for r in rows if r["wls"][0] <= 2.0 and r["wls"][1] <= 15.0]

    def ok(e):
        return e is not None and e[0] <= 1.0 and e[1] <= 5.0
    defined = [r for r in inmode if r["oracle"] is not None]
    missed = [r for r in inmode if not ok(r["wls"])]
    return dict(in_mode=len(inmode), oracle_defined=len(defined),
                share_within_tolerance_where_oracle_defined={k: float(np.mean([ok(r[k]) for r in defined])) for k in ("wls", "r3", "oracle")},
                wls_out_of_tolerance=len(missed), fixed_by_r3=sum(ok(r["r3"]) for r in missed), fixed_by_oracle=sum(ok(r["oracle"]) for r in missed),
                clique_size_median=float(np.median([r["n"] for r in inmode])))


variants = {}
for v, d in RUN_DIRS.items():
    reps = sorted(d.glob("rep*"))
    rows = []
    for r in reps:
        s = json.loads((r / "run_summary.json").read_text())
        assert s["replay_mismatches"] == 0, r
        row = dict(decompose(r / "error_traces.json.gz"), availability=s["availability"],
                   translation_m=s["mean_of_pair_means_translation_m"], yaw_deg=s["mean_of_pair_means_yaw_deg"],
                   first_acceptance_median_s=float(np.median(list(s["first_acceptance_s"].values()))))
        timing = r / "timing.json"
        if timing.exists():
            t = json.loads(timing.read_text())
            row["manager_seconds"] = t["seconds"]
        row["per_pair_acc"] = per_pair(r / "error_traces.json.gz")
        rows.append(row)
    keys = [k for k in rows[0] if k != "per_pair_acc"]
    variants[v] = dict(repeats=len(rows), accepted_error_profile=error_profile([r / "error_traces.json.gz" for r in reps]),
                       mean={k: float(np.mean([r[k] for r in rows])) for k in keys},
                       range={k: float(np.ptp([r[k] for r in rows])) for k in keys},
                       per_pair={p: {k: (float(np.mean([r["per_pair_acc"][p][k] for r in rows])) if k != "longest_yaw_gt10_run"
                                         else [r["per_pair_acc"][p][k] for r in rows]) for k in rows[0]["per_pair_acc"][p]}
                                 for p in rows[0]["per_pair_acc"]})

cold = {v: json.loads((RES / "cold_start" / f"{v}.json").read_text()) for v in ("B0", "G2", "S2", "R3")}
def wrong_first_by_window(c):
    out = {}
    for r in c["runs"]:
        m = r["metrics"]
        if m["first_solution"] == "wrong":
            out.setdefault(f"{r['episode_id']} {r['pair']}", []).append([m["median_translation_m"], m["median_yaw_deg"]])
    return {k: dict(starts=len(x), window_median_translation_m=float(np.median([a for a, _ in x])),
                    window_median_yaw_deg=float(np.median([b for _, b in x]))) for k, x in out.items()}


cold_view = {v: {k: c[k] for k in ("starts", "wrong_first", "no_first", "correct_by_horizon", "capped_time_to_correct_s",
                                   "correct_update_availability", "mean_runtime_per_window_s")}
             | {"wrong_first_by_window": wrong_first_by_window(c)} for v, c in cold.items()}
t20 = json.loads((RES / "size_auc.json").read_text())
t30 = json.loads((RES / "robust_offline.json").read_text())
t0 = json.loads((RES / "coverage.json").read_text())


def exceeds(a, b, key, sign=1):
    diff = sign * (variants[a]["mean"][key] - variants[b]["mean"][key])
    return diff, max(variants[a]["range"][key], variants[b]["range"][key])


h2 = {}
h2["T2.0_passed"] = bool(t20["passed"])
d1, r1 = exceeds("S2", "B0", "acc")
d2, r2 = exceeds("S2", "G2", "acc")
h2["acc_S2_minus_B0"], h2["range_S2_B0"] = d1, r1
h2["acc_S2_minus_G2"], h2["range_S2_G2"] = d2, r2
h2["acc_condition"] = bool(d1 > r1 and d2 > r2)
h2["cold_condition"] = bool(cold["S2"]["wrong_first"] <= cold["B0"]["wrong_first"] and cold["S2"]["no_first"] < cold["B0"]["no_first"])
h2["runtime_condition"] = bool(cold["S2"]["mean_runtime_per_window_s"] <= 1.25 * cold["G2"]["mean_runtime_per_window_s"])
h2["supported"] = all(h2[k] for k in ("T2.0_passed", "acc_condition", "cold_condition", "runtime_condition"))

h3 = {}
h3["T3.0_passed"] = bool(t30["passed"])
d1, r1 = exceeds("R3", "B0", "acc")
d2, r2 = exceeds("B0", "R3", "e_sum")
h3["acc_R3_minus_B0"], h3["range_R3_B0"] = d1, r1
h3["error_share_B0_minus_R3"], h3["range_error_share"] = d2, r2
h3["acc_condition"] = bool(d1 > r1)
h3["error_condition"] = bool(d2 > r2)
h3["u_pre_condition"] = bool(variants["R3"]["mean"]["u_pre"] - variants["B0"]["mean"]["u_pre"] <= 0.01)
h3["cold_condition"] = bool(cold["R3"]["wrong_first"] <= cold["B0"]["wrong_first"])
h3["supported"] = all(h3[k] for k in ("T3.0_passed", "acc_condition", "error_condition", "u_pre_condition", "cold_condition"))

out = {"protocol": "configs/support_hypotheses.json", "full_recording": variants, "cold_start": cold_view,
       "T0": {k: t0[k] for k in ("share_pre_steps_with_measurement", "share_pre_steps_with_correct_measurement",
                                 "lag_first_correct_measurement_to_acceptance_steps", "pre_steps_with_gt")},
       "T2.0": {k: t20[k] for k in ("all", "gate_rejected", "passed")},
       "T3.0": {k: t30[k] for k in ("in_mode_measurements", "share_within_tolerance_in_mode", "paired_difference_r3_minus_wls",
                                    "median_error_in_mode", "residual_auc_oracle_membership", "passed")},
       "T3.0_oracle_detail": oracle_detail(RES / "robust_offline.rows.json.gz"),
       "H2": h2, "H3": h3}
(RES / "aggregate.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"H2": h2, "H3": h3, "cold_start": cold_view,
                  "full_recording": {v: {k: round(x, 4) for k, x in d["mean"].items()} for v, d in variants.items()}}, indent=1))
