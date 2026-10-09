#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TCAFF_MOT_DATASET="$PWD/data/tcaff_mot_data"
export PYTHONPATH="$PWD/vendor/tcaff"
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tcaff-mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
output_dir="${2:-results/demo_smoke}"
mkdir -p "$output_dir" "$MPLCONFIGDIR"
.venv/bin/python src/demo_cpu/demo.py --params "${1:-configs/demo_smoke.yaml}" --output "$output_dir/alignment.png"
