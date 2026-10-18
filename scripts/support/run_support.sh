#!/usr/bin/env bash
# Full-recording runs of the H2/H3 protocol on the authors' detections, REPEATS times, PARALLEL jobs at once.
# Usage: run_support.sh LAB_ROOT OUT_DIR REPEATS PARALLEL   (VARIANTS env, default: G2 S2 R3)
# Use OUT_DIR=<results>/runs so that aggregate.py <results> finds OUT_DIR/{G2,S2,R3}/rep*.
# Same harness as scripts/segment_unit/run_variants.sh: capture of candidates and maps, then replay verification.
set -euo pipefail
LAB="$1"; OUT="$2"; REPEATS="${3:-5}"; PAR="${4:-4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DET="$LAB/data/tcaff_mot_data/fastsam_data"
jobs=()
for k in $(seq 1 "$REPEATS"); do
  for V in ${VARIANTS:-G2 S2 R3}; do jobs+=("$V $k"); done
done
run_one() {
  V="$1"; k="$2"
  RUN="$OUT/$V/rep$k"; CAP="$RUN/capture"
  [[ -f "$RUN/run_summary.json" ]] && exit 0
  mkdir -p "$RUN"
  case "$V" in G2|S2) GATE=', "tcaff": {"wh_scale_diff": 1.7, "h_diff": 0.2}';; *) GATE='';; esac
  cd "$LAB"
  LAB_ROOT="$LAB" TCAFF_MOT_DATASET="$LAB/data/tcaff_mot_data" PYTHONPATH="$LAB/vendor/tcaff" \
  CAPTURE_CANDIDATES=1 CAPTURE_MAPS=1 CAPTURE_DIR="$CAP" SUPPORT_TIMING_FILE="$RUN/timing.json" \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tcaff-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$LAB/.venv/bin/python" "$HERE/run_demo.py" "$V" --params vendor/tcaff/demo/params/tcaff_mot_dataset.yaml \
    --output "$RUN/alignment.png" -x "{\"fastsam3d_data\": {\"root\": \"$DET\"}$GATE${EXTRA_OVERRIDE:-}}" > "$RUN/run.log" 2>&1
  # G2/S2 change the size gate on purpose: their evaluator allows only that difference from the released YAML.
  EVAL="$HERE/../segment_unit/evaluate_run.py"; case "$V" in G2|S2) EVAL="$HERE/evaluate_gate_run.py";; esac
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$LAB/.venv/bin/python" -I "$EVAL" \
    "$LAB" "$CAP" "$RUN" "$DET" > "$RUN/evaluate.log" 2>&1
  echo "$V rep$k $(tail -1 "$RUN/evaluate.log")"
}
export -f run_one; export LAB OUT HERE DET
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c 'run_one $0 $1'
echo "ALL RUNS DONE"
