#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_name="tcaff_baseline_$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="${1:-results/$run_name}"
capture_dir="${2:-data/$run_name}"
if [[ -e "$run_dir" || -e "$capture_dir" ]]; then
  echo "Choose new output directories to preserve the frozen baseline and its saved proposals." >&2
  exit 1
fi
mkdir -p "$run_dir"
export CAPTURE_CANDIDATES=1 CAPTURE_MAPS=1 CAPTURE_DIR="$capture_dir"
export MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tcaff-mpl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
echo "Running released TCAFF; log: $run_dir/run.log"
bash scripts/run_demo.sh vendor/tcaff/demo/params/tcaff_mot_dataset.yaml "$run_dir" > "$run_dir/run.log" 2>&1
.venv/bin/python scripts/tcaff_baseline_audit.py --input "$capture_dir" --output "$run_dir" > "$run_dir/audit.log" 2>&1
echo "Baseline and replay verified: $run_dir/audit.json"
