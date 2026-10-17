"""Replay-verify one TCAFF run on variant detections and compute its metrics.

Reuses replay() and summarize() of the baseline audit (filter replay must match the run,
GT is recomposed independently). Records hashes of the variant detection files actually used.

Usage: python -I evaluate_run.py LAB_ROOT CAPTURE_DIR RUN_DIR VARIANT_DIR
"""
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

LAB, CAP, RUN, VDIR = (Path(a) for a in sys.argv[1:5])
os.environ["TCAFF_MOT_DATASET"] = str(LAB / "data/tcaff_mot_data")  # as in the audit's main()
sys.path.insert(0, str(LAB / "scripts"))
import tcaff_baseline_audit as audit  # noqa: E402

params = yaml.safe_load((LAB / "vendor/tcaff/demo/params/tcaff_mot_dataset.yaml").read_text())
poses = audit.load_poses(params)
paths = sorted(CAP.glob("*.json"))
assert len(paths) == 12, "Expected all 12 directed pairs"
traces = [audit.replay(json.loads(p.read_text()), params, poses) for p in paths]
s = audit.summarize(traces, "release_tau9")
frames = [f["release_tau9"] for t in traces for f in t["frames"]]
correct = sum(1 for f in frames if f["translation_m"] is not None and f["translation_m"] <= 1.0 and f["yaw_deg"] <= 5.0)
result = {"availability": s["availability"],
          "mean_of_pair_means_translation_m": s["author_mean_of_pair_means_translation_m"],
          "mean_of_pair_means_yaw_deg": s["author_mean_of_pair_means_yaw_deg"],
          "pooled_mean_translation_m": s["pooled_mean_translation_m"],
          "correct_share_all_updates": correct / s["pair_updates"],
          "correct_updates": correct, "pair_updates": s["pair_updates"],
          "first_acceptance_s": {p["pair"]: p["first_acceptance_s"] for p in s["pairs"]},
          "replay_mismatches": sum(len(t["replay_mismatches"]) for t in traces),
          "max_replay_coordinate_error": max(t["max_replay_coordinate_error"] for t in traces),
          "variant_inputs": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(VDIR.glob("RR*.json"))]}
RUN.mkdir(parents=True, exist_ok=True)
(RUN / "run_summary.json").write_text(json.dumps(result, indent=1))
with gzip.open(RUN / "error_traces.json.gz", "wt") as f:
    json.dump([{"pair": t["pair"], "frames": [[x["step"], x["release_tau9"]["accepted"], x["release_tau9"]["translation_m"], x["release_tau9"]["yaw_deg"]] for x in t["frames"]]} for t in traces], f)
print(json.dumps({k: v for k, v in result.items() if k not in ("first_acceptance_s", "variant_inputs")}))
