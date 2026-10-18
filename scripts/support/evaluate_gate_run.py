"""Evaluate a G2/S2 run with scripts/segment_unit/evaluate_run.py.

The audit replay asserts that captured TCAFF parameters equal the released YAML. G2/S2 change the
size gate on purpose, so here the captured parameters may differ only in wh_scale_diff = 1.7 and
h_diff = 0.2 (protocol values); every other parameter must match. The temporal-filter replay and the
independent GT composition then run unchanged.

Usage: python -I evaluate_gate_run.py LAB_ROOT CAPTURE_DIR RUN_DIR DETECTION_DIR
"""
import copy
import runpy
import sys
from pathlib import Path

LAB = Path(sys.argv[1])
sys.path.insert(0, str(LAB / "scripts"))
import tcaff_baseline_audit as audit  # noqa: E402

GATE = {"wh_scale_diff": 1.7, "h_diff": 0.2}
_replay = audit.replay


def replay(data, params, poses):
    captured, released = data["params"], params["tcaff"]
    assert set(captured) == set(released), "Captured parameter keys differ"
    differing = {k for k in released if captured[k] != released[k]}
    assert differing <= set(GATE) and all(abs(captured[k] - GATE[k]) < 1e-12 for k in differing), differing
    patched = copy.deepcopy(params)
    patched["tcaff"] = captured
    return _replay(data, patched, poses)


audit.replay = replay
script = Path(__file__).resolve().parents[1] / "segment_unit/evaluate_run.py"
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
