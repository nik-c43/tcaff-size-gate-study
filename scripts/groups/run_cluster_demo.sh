#!/usr/bin/env bash
# Full demo runs with cluster_patch, REPEATS in parallel, each verified by the repository's evaluate_run.py.
# Usage: run_cluster_demo.sh LAB_ROOT OUT_DIR REPEATS [PARAMS_YAML]
set -euo pipefail
LAB="$1"; REPEATS="$3"; PARAMS="${4:-vendor/tcaff/demo/params/tcaff_mot_dataset.yaml}"
OUT="$(mkdir -p "$2" && cd "$2" && pwd)"   # absolute: run_one changes into LAB
HERE="$(cd "$(dirname "$0")" && pwd)"
DET="$LAB/data/tcaff_mot_data/fastsam_data"
EVAL="$HERE/../segment_unit/evaluate_run.py"
run_one() {
  k="$1"; RUN="$OUT/KBWA/rep$k"; CAP="$RUN/capture"
  [[ -f "$RUN/run_summary.json" ]] && exit 0
  mkdir -p "$RUN"; cd "$LAB"
  LAB_ROOT="$LAB" TCAFF_MOT_DATASET="$LAB/data/tcaff_mot_data" PYTHONPATH="$LAB/vendor/tcaff" \
  CAPTURE_CANDIDATES=1 CAPTURE_MAPS=1 CAPTURE_DIR="$CAP" \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/tcaff-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$LAB/.venv/bin/python" "$HERE/run_cluster_demo.py" --params "$PARAMS" \
    --output "$RUN/alignment.png" -x "{\"fastsam3d_data\": {\"root\": \"$DET\"}}" > "$RUN/run.log" 2>&1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$LAB/.venv/bin/python" -I "$EVAL" "$LAB" "$CAP" "$RUN" "$DET" > "$RUN/evaluate.log" 2>&1
  echo "rep$k $(tail -1 "$RUN/evaluate.log")"
}
export -f run_one; export LAB OUT HERE DET EVAL PARAMS
seq 1 "$REPEATS" | xargs -P "$REPEATS" -I{} bash -c 'run_one {}'
done_runs=$(ls "$OUT"/KBWA/rep*/run_summary.json 2>/dev/null | wc -l)
if [[ "$done_runs" -lt "$REPEATS" ]]; then echo "only $done_runs of $REPEATS runs finished" >&2; exit 1; fi
echo "ALL RUNS DONE"
