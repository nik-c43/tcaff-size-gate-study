#!/usr/bin/env bash
# FastSAM frontend on every frame of the four robots: extract frames, segment, delete frames.
# Usage: run_all.sh WORKSPACE_ROOT OUT_DIR FASTSAM_X_WEIGHTS
# The executed copy differed only in taking these three paths as constants.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
LAB="${1:?usage: run_all.sh WORKSPACE_ROOT OUT_DIR WEIGHTS}"
OUT="${2:?usage: run_all.sh WORKSPACE_ROOT OUT_DIR WEIGHTS}"
WEIGHTS="${3:?usage: run_all.sh WORKSPACE_ROOT OUT_DIR WEIGHTS}"
for R in RR01 RR04 RR06 RR08; do
  if [[ ! -f "$OUT/$R/manifest.json" ]]; then
    "$LAB/.venv/bin/python" -I "$S/extract_all.py" "$LAB" "$OUT" "$R" 2>&1 | grep -v cv_bridge
  fi
  "$LAB/.venv-vision/bin/python" -I "$S/segment_all.py" "$WEIGHTS" "$OUT/$R" \
     "$LAB/data/tcaff_mot_data/objects.txt" 2>&1 | grep -v -i 'warn\|torch.load'
  rm -rf "$OUT/$R/frames"
done
echo "ALL DONE"
