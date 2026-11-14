# Implementation and validation notes

## 14 November 2026 — implementation milestone

Work was executed with Codex on the development Mac (Apple Silicon), using a
Python 3.11 virtual environment. Package versions, configuration, platform,
source-code SHA-256, runtime, and Git revision are retained in run JSON files.
No student contribution claims or experiment values have been invented.

### Implemented

- From-scratch normalized eight-point geometry, adaptive RANSAC, essential
  decomposition, cheirality, n-view triangulation, DLT PnP, and union-find tracks.
- Corrected sparse BA indexing for multiple free cameras. Fixed the first camera
  and constrained one baseline length so the second pose can refine.
- Homography/parallax initialization checks, failed-candidate retry, local and
  periodic/global BA, positive-depth and point filtering, and growth exports.
- SIFT/RootSIFT/ORB; content-addressed feature caching; configurable matching,
  initialization, triangulation, registration, and optimization.
- EXIF preparation, calibrated K preservation, camera/filename conversion alongside
  image resizing, checkerboard calibration and undistortion path.
- Safe Middlebury archive loading with camera conversion and SHA-256 provenance.
- CPU pycolmap 4.2.1 baseline with fixed shared K and largest-model export.
- VGGT CUDA adapter, Colab notebook, and common NPZ conversion. GPU inference
  has **not** been executed locally. Conversion is tested with known transforms.
- Sim(3), ATE, relative-pose AUC, registration rate, reprojection statistics,
  optional density-matched cloud metrics, CSV/LaTeX tables, PNG/orbit rendering,
  and logged BA/detector/view-count experiment runners.
- Original eight-view generated image dataset with exact camera poses. A measured
  COLMAP baseline for that generated dataset is included in `sample_data`.

### Validation and development findings

`python -m pytest -q` passed 13 tests, including exact/noisy geometry, 30% match
outliers, BA with multiple free cameras, alignment, missing-registration penalties,
zero-baseline reference pairs, preparation/conversions, the actual image pipeline
from a different working directory, and a rejected PnP candidate followed by retry.
`pip check`, compilation, and Git whitespace checks passed.

The first default Temple run registered 5/16 views. Lowering the SIFT threshold
and using RootSIFT alone registered 4/16. Freeing the second pose greatly reduced
residuals but did not solve coverage by itself. The explicitly named
`configs/sfm_relaxed.yaml` also relaxes matching, triangulation, and registration
thresholds, and reconstructed all 16 views. This tuning is scene-dependent and
must not be presented as a universal improvement. Default settings remain useful
for the generated demo. Development runs are retained locally in `outputs/` and
`results/`; they are not silently replaced with more flattering numbers.

A real low-inlier pair exposed cancellation in `log(1 - w**8)`, producing division
by zero in adaptive RANSAC. It now uses `log1p`, with a regression test for very
low inlier fractions. A Middlebury request initially returned HTTP 403; a browser
User-Agent allowed the official archive to download. No mirror or undocumented
camera convention was substituted.

The Temple calibration follows the official `K[R|t]` file format:
https://vision.middlebury.edu/mview/data/
The GT comparison to COLMAP also provides an independent pose-convention check.
The dataset remains outside the repository. Original generated images have no
third-party image redistribution dependency.

### Reproduction commands

Activate `.venv` and run the README commands. The following are the principal
validation runs (full summaries are under `validation/`):

```bash
python -m pytest -q
bash code/run_all.sh
python code/run_sfm.py --images data/temple/images --out outputs/temple_relaxed_final --config configs/sfm_relaxed.yaml
python code/baselines/run_colmap.py --images data/temple/images --out outputs/temple_colmap_final
python code/run_experiments.py --scene sample_data/scene_small --out results/sample_final_ba --ablation ba
python code/run_experiments.py --scene sample_data/scene_small --out results/sample_views --ablation views --views 5 8 --seed 0
```

Generated-sample BA modes and 5/8-view subsets were run during development. These
are smoke experiments, not the full real-scene E1/E2/E3 study. Development logs
may reference the pre-implementation parent revision; the source fingerprint and
embedded configuration distinguish them. Final smoke results reference the
implementation commit. The fresh-source check uses a ZIP of tracked source and
the existing validated Python environment; it is **not** a fresh dependency
installation or a completed submission-archive acceptance test.

### Remaining work

1. VGGT inference on a GPU and round-trip checks against actual model output.
2. Multiple real scenes and failure cases; the complete view-count, BA, detector,
   and calibration experiment matrix with consistent subsets and hardware notes.
3. Validate real checkerboard calibration/undistortion with captured images.
4. Complete scientific figures, the 4–6 page IEEE report, and the 1–2 minute demo.
   The generated orbit preview is a short rendering check, not the final demo.
5. Group number, real author details, and actual student contribution evidence.
6. Final zip under 200 MB and a fresh-unzip installation/run check. Portal upload
   and any account-based Colab/Overleaf work still require the team.

Dense reconstruction, meshes, neural rendering, loop closure, and multicomponent
merging remain outside the implemented sparse core. The maximum group size in
the embedded assignment is four; five workstreams are not five members.

### Retained measurements

| Logged run | Registered | ATE / GT extent | AUC@5 |
|---|---:|---:|---:|
| sample_ours.json | 8/8 | 0.047% | 0.982 |
| sample_colmap.json | 8/8 | 0.040% | 0.983 |
| temple_ours.json | 16/16 | 0.433% | 0.766 |
| temple_colmap.json | 13/16 | 0.081% | 0.612 |

See `validation/README.md` for interpretation and source metadata.
