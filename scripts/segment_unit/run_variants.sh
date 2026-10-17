#!/usr/bin/env bash
# Run the released TCAFF demo on each variant's detections, REPEATS times, PARALLEL jobs at once.
# Usage: run_variants.sh LAB_ROOT VARIANTS_DIR OUT_DIR REPEATS PARALLEL
# VARIANTS env selects variant subdirectories (default: R0 U1 U2 U3; B0 = reference copy of the authors' detections).
set -euo pipefail
LAB="$1"; VARS="$2"; OUT="$3"; REPEATS="${4:-5}"; PAR="${5:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
jobs=()
for k in $(seq 1 "$REPEATS"); do
  for V in ${VARIANTS:-R0 U1 U2 U3}; do jobs+=("$V $k"); done
done
run_one() {
  V="$1"; k="$2"
  RUN="$OUT/$V/rep$k"; CAP="$RUN/capture"
  [[ -f "$RUN/run_summary.json" ]] && exit 0
  mkdir -p "$RUN"
  cd "$LAB"
  TCAFF_MOT_DATASET="$LAB/data/tcaff_mot_data" PYTHONPATH="$LAB/vendor/tcaff" \
  CAPTURE_CANDIDATES=1 CAPTURE_MAPS=1 CAPTURE_DIR="$CAP" \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tcaff-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$LAB/.venv/bin/python" src/demo_cpu/demo.py --params vendor/tcaff/demo/params/tcaff_mot_dataset.yaml \
    --output "$RUN/alignment.png" -x "{\"fastsam3d_data\": {\"root\": \"$VARS/$V\"}}" > "$RUN/run.log" 2>&1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$LAB/.venv/bin/python" -I "$HERE/evaluate_run.py" \
    "$LAB" "$CAP" "$RUN" "$VARS/$V" > "$RUN/evaluate.log" 2>&1
  echo "$V rep$k $(tail -1 "$RUN/evaluate.log")"
}
export -f run_one; export LAB VARS OUT HERE
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c 'run_one $0 $1'
echo "ALL RUNS DONE"
