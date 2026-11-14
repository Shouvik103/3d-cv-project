# Classical vs. Learned Structure from Motion

This project implements an incremental sparse Structure-from-Motion (SfM) pipeline
from first principles and benchmarks it against COLMAP and VGGT. It uses public,
calibrated Middlebury scenes for reproducible pose accuracy and can also process
self-captured image sets.

## Five-workstream layout

The repository can be run on one machine, but its interfaces are deliberately
separated so each of five team members can make a real, reviewable contribution.

| Workstream | Responsibility | Primary locations |
|---|---|---|
| A — Data and calibration | Dataset download, image preparation, GT conversion, calibration | `code/data/`, `code/prepare_data.py` |
| B — Two-view geometry | Features, matching, F/E, RANSAC, triangulation | `code/sfm/features.py`, `matching.py`, `geometry.py` |
| C — Incremental reconstruction | Tracks, PnP registration, bundle adjustment, exports | `code/sfm/tracks.py`, `incremental.py`, `bundle_adjust.py` |
| D — Baselines | COLMAP/VGGT adapters and common pose conversion | `code/baselines/` |
| E — Evaluation and communication | Metrics, tables, figures, report/demo assets | `code/eval/`, `code/viz/`, `report/` |

`TEAM_CONTRIBUTIONS.md` is a template: record only work that a person actually
performs, reviews, or validates.

## Install

Use Python 3.11 on Apple Silicon:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python code/data/download_middlebury.py --scene templeSparseRing --out data/middlebury
python code/prepare_data.py --images data/middlebury/templeSparseRing --out data/temple/images_1600
python code/run_sfm.py --images data/temple/images_1600 --out outputs/temple_ours --config configs/sfm_default.yaml
pytest -q
```

COLMAP and VGGT are optional baselines; their wrappers are intentionally kept
separate from the from-scratch geometry implementation.

## Pose convention

All methods export `poses.json` using OpenCV world-to-camera poses:
`x_cam = R @ X_world + t`, with camera center `C = -R.T @ t`.

## Dataset provenance

The downloader targets the public Middlebury multiview `templeSparseRing` or
`dinoSparseRing` archives. Do not package full datasets or model weights in the
submission zip; retain only a small permitted sample input.
