"""Run the CPU demo of TCAFF with cluster_patch applied. Usage: LAB_ROOT=... python run_cluster_demo.py <demo.py arguments>"""
import os
import runpy
import sys
from pathlib import Path

LAB = Path(os.environ["LAB_ROOT"])
sys.path[:0] = [str(LAB / "vendor/tcaff"), str(LAB / "src/demo_cpu"), str(Path(__file__).resolve().parent)]
import cluster_patch  # noqa: E402

cluster_patch.apply()
sys.argv = [str(LAB / "src/demo_cpu/demo.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
