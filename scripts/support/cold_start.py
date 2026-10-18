"""Cold starts of the H2/H3 protocol on the 12 holdout windows of report section 6.4.

Runs run_episode of scripts/size_gate_evaluate.py unchanged on the frozen author maps, with the
variant applied by patches.py (G2/S2: gate alpha = 2; B0/R3: alpha = 1). Filter replay is verified
inside run_episode. Only per-run metrics are stored.

Usage: python -I cold_start.py LAB_ROOT VARIANT OUT_JSON [REPEATS]
"""
import json
import sys
from pathlib import Path

LAB, VARIANT, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
sys.path[:0] = [str(LAB / "scripts"), str(LAB / "vendor/tcaff"), str(Path(__file__).resolve().parent)]
import patches  # noqa: E402

patches.apply(VARIANT)
import size_gate_evaluate as sge  # noqa: E402
from tcaff_baseline_audit import digest  # noqa: E402

manifest = json.loads((LAB / "data/size_gate_study/analysis_private/manifest.json").read_text())
episodes = sorted([e for e in manifest["episodes"] if e["split"] == "holdout"], key=lambda e: (e["start"], e["pair"]))
assert len(episodes) == 12
source = {}
for pair in {e["pair"] for e in episodes}:
    p = LAB / manifest["protocol"]["baseline"] / f"{pair}.json"
    assert digest(p) == manifest["provenance"]["source_hashes"][pair]["sha256"], pair
    source[pair] = json.loads(p.read_text())
alpha = 2.0 if VARIANT in ("G2", "S2") else 1.0
runs = []
for e in episodes:
    for rep in range(1, REPEATS + 1):
        run = sge.run_episode(source[e["pair"]], e, alpha, rep)
        runs.append(dict(episode_id=run["episode_id"], pair=run["pair"], repetition=rep, metrics=run["metrics"]))
        print(VARIANT, e["episode_id"], rep, run["metrics"]["first_solution"], flush=True)
summary = sge.aggregate(runs)
n = len(runs)
res = {"test": "cold start, 12 holdout windows of section 6.4 (configs/support_hypotheses.json)", "variant": VARIANT, "alpha": alpha,
       "starts": n,
       "wrong_first": sum(r["metrics"]["first_solution"] == "wrong" for r in runs),
       "no_first": sum(r["metrics"]["first_solution"] == "none" for r in runs),
       "correct_by_horizon": sum(r["metrics"]["correctly_initialized_by_horizon"] for r in runs),
       "capped_time_to_correct_s": summary["capped_time_to_correct_s"],
       "correct_update_availability": summary["correct_update_availability"],
       "mean_runtime_per_window_s": summary["runtime_s"],
       "summary": summary, "runs": runs}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ("summary", "runs")}))
