"""T0: before the first acceptance, is a correct MNO measurement already available? (protocol T0)

For each B0 capture (directed pair, repeat): steps before the first accepted estimate that have GT;
share with at least one measurement and with at least one measurement within 1 m and 5 deg of GT;
and the lag from the first correct measurement to the first acceptance.

Usage: python -I coverage.py LAB_ROOT OUT_JSON CAPTURE_DIR [CAPTURE_DIR ...]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LAB, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff")]
from tcaff_baseline_audit import pose_error  # noqa: E402

rows = []
for rep, cap in enumerate(sys.argv[3:], 1):
    for path in sorted(Path(cap).glob("RR*_RR*.json")):
        frames = json.loads(path.read_text())["frames"]
        first = next((i for i, f in enumerate(frames) if f["baseline_estimate"] is not None), None)
        pre = frames[:first] if first is not None else frames
        with_gt = [f for f in pre if f["truth"] is not None]
        ok = [any(e[0] <= 1.0 and e[1] <= 5.0 for e in (pose_error(z, f["truth"]) for z in f["zs"])) for f in with_gt]
        first_ok = next((f["step"] for f, o in zip(with_gt, ok) if o), None)
        rows.append(dict(pair=path.stem, repeat=rep, first_acceptance_step=None if first is None else frames[first]["step"],
                         pre_steps_with_gt=len(with_gt), pre_steps_with_measurement=sum(bool(f["zs"]) for f in with_gt),
                         pre_steps_with_correct_measurement=sum(ok), first_correct_measurement_step=first_ok,
                         lag_first_correct_to_acceptance=None if first_ok is None or first is None else frames[first]["step"] - first_ok))

total = {k: sum(r[k] for r in rows) for k in ("pre_steps_with_gt", "pre_steps_with_measurement", "pre_steps_with_correct_measurement")}
lags = [r["lag_first_correct_to_acceptance"] for r in rows if r["lag_first_correct_to_acceptance"] is not None]
per_pair = defaultdict(list)
for r in rows:
    per_pair[r["pair"]].append(r)
res = {"test": "T0 coverage before first acceptance (configs/support_hypotheses.json)",
       "captures": len(rows),
       "share_pre_steps_with_measurement": total["pre_steps_with_measurement"] / total["pre_steps_with_gt"],
       "share_pre_steps_with_correct_measurement": total["pre_steps_with_correct_measurement"] / total["pre_steps_with_gt"],
       "pre_steps_with_gt": total["pre_steps_with_gt"],
       "lag_first_correct_measurement_to_acceptance_steps": {"median": float(np.median(lags)), "q25_q75": np.quantile(lags, [.25, .75]).tolist(), "min_max": [min(lags), max(lags)], "n": len(lags)},
       "per_pair": {p: {"first_acceptance_step_median": float(np.median([r["first_acceptance_step"] for r in rs])),
                        "first_correct_measurement_step_median": float(np.median([r["first_correct_measurement_step"] for r in rs if r["first_correct_measurement_step"] is not None])),
                        "share_pre_steps_with_correct_measurement": sum(r["pre_steps_with_correct_measurement"] for r in rs) / max(1, sum(r["pre_steps_with_gt"] for r in rs))}
                    for p, rs in sorted(per_pair.items())},
       "rows": rows}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("rows",)}, indent=1))
