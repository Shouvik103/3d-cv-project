#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
SCENE="${1:-sample_data/scene_small}"
TAG="${2:-sample}"
"$PYTHON" code/run_sfm.py --images "$SCENE/images" --out "outputs/${TAG}_ours" --config configs/sfm_default.yaml
"$PYTHON" code/eval/run_eval.py --estimate "outputs/${TAG}_ours" --reference "$SCENE/gt_poses.json" --reference-kind gt --out "results/$TAG/ours.json"
"$PYTHON" code/viz/render_orbit.py --reconstruction "outputs/${TAG}_ours" --out "outputs/${TAG}_ours/preview.png"
if "$PYTHON" -c 'import pycolmap' >/dev/null 2>&1; then
  if [ ! -f "outputs/${TAG}_colmap/summary.json" ]; then
    "$PYTHON" code/baselines/run_colmap.py --images "$SCENE/images" --out "outputs/${TAG}_colmap"
  fi
  "$PYTHON" code/eval/run_eval.py --estimate "outputs/${TAG}_colmap" --reference "$SCENE/gt_poses.json" --reference-kind gt --out "results/$TAG/colmap.json"
fi
"$PYTHON" code/eval/make_tables.py --results "results/$TAG" --out "report/tables/$TAG"
