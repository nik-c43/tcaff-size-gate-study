"""Run the CPU demo of TCAFF with one protocol variant applied (patches.py).

Usage: LAB_ROOT=... python run_demo.py VARIANT <demo.py arguments>
"""
import os
import runpy
import sys
from pathlib import Path

LAB = Path(os.environ["LAB_ROOT"])
sys.path[:0] = [str(LAB / "vendor/tcaff"), str(LAB / "src/demo_cpu"), str(Path(__file__).resolve().parent)]
import patches  # noqa: E402

patches.apply(sys.argv[1])
sys.argv = [str(LAB / "src/demo_cpu/demo.py")] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
