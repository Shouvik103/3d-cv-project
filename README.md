# Classical vs. Learned Structure from Motion

Incremental sparse Structure from Motion from overlapping images, compared with
CPU COLMAP and a VGGT inference adapter. The geometric implementation includes
normalized eight-point estimation, adaptive RANSAC, essential-matrix pose recovery,
multiview triangulation, feature tracks, PnP registration, and sparse robust bundle
adjustment. OpenCV supplies SIFT/ORB, matching, homography detection, and PnP;
SciPy supplies the optimizer. This is an implementation in progress, not a final
course submission.

## Install

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional CPU COLMAP baseline:
pip install -r requirements-colmap.txt
```

Dependencies are pinned to the versions used for local validation. VGGT's torch,
GPU libraries, and weights are installed separately on Colab. The core demo needs
no GPU, model weights, Homebrew tools, or dataset download.

## Quick demo

Eight original synthetic views, exact intrinsics, and ground-truth poses are
included under `sample_data/scene_small/`. The images show textured tiles at
multiple depths. This is a controlled smoke test, not evidence of real-world
reconstruction quality.

```bash
python code/run_sfm.py --images sample_data/scene_small/images --out outputs/sample_ours --config configs/sfm_default.yaml
python code/eval/run_eval.py --estimate outputs/sample_ours --reference sample_data/scene_small/gt_poses.json --reference-kind gt --out results/sample/ours.json
python code/viz/render_orbit.py --reconstruction outputs/sample_ours --out outputs/sample_ours/preview.png
python -m pytest -q
```

Expected outputs: `poses.json`, colored `points.ply`, `observations.json`,
`summary.json` with measured reprojection statistics and runtime, `run.log`,
growth snapshots, an evaluation JSON, and a PNG preview. The sample typically
takes roughly 10–20 seconds on the development Mac; timings vary by hardware.
The test suite also invokes the actual image-based command from another working
directory and checks pose accuracy against ground truth.

`bash code/run_all.sh` runs reconstruction, evaluation, preview, optional COLMAP,
and table generation on this sample. Set `PYTHON=.venv/bin/python` if the virtual
environment is not activated. COLMAP requires a fresh output directory for each
new run. Existing successful COLMAP output is reused by `run_all.sh`.

## Public calibrated scene

```bash
python code/data/download_middlebury.py --scene templeSparseRing --out data/middlebury
python code/prepare_data.py --images data/middlebury/templeSparseRing --out data/temple/images --intrinsics data/middlebury/templeSparseRing/intrinsics.yaml --poses data/middlebury/templeSparseRing/gt_poses.json
python code/run_sfm.py --images data/temple/images --out outputs/temple_ours --config configs/sfm_relaxed.yaml
python code/baselines/run_colmap.py --images data/temple/images --out outputs/temple_colmap
python code/eval/run_eval.py --estimate outputs/temple_ours --reference data/temple/gt_poses.json --reference-kind gt --out results/temple/ours.json
```

The loader preserves the dataset's calibrated `K[R|t]` convention. Preparation
renames images and ground-truth entries together and scales intrinsics. Omitting
`--intrinsics` estimates focal length from EXIF or uses `1.2 * max(width,height)`;
that estimate must not replace known calibration during a ground-truth benchmark.
Archives are cached with SHA-256 provenance. Download failures remain explicit.
See the [Middlebury dataset and format description](https://vision.middlebury.edu/mview/data/).
Do not redistribute the downloaded dataset without checking its terms.

## Your own photos and calibration

Use a static textured scene, one camera/lens, sharp photos, and substantial overlap.

```bash
python code/prepare_data.py --images data/my_scene/raw --out data/my_scene/images
python code/run_sfm.py --images data/my_scene/images --out outputs/my_scene --config configs/sfm_default.yaml
```

JPEG, PNG, and TIFF preparation are supported. Convert HEIC to JPEG first. EXIF
orientation is applied before resizing. An optional checkerboard calibration can
be produced and applied at the same camera resolution:

```bash
python code/calibrate_camera.py --images data/checkerboard --board 9 6 --square-size 25 --out data/calibration.yaml
python code/prepare_data.py --images data/my_scene/raw --out data/my_scene/images --calibration data/calibration.yaml
```

This path undistorts images before resizing. Calibration requires at least five
detected boards; more varied views are preferable. Real checkerboard calibration
has not been validated in this repository yet.

## Baselines and experiments

COLMAP uses the same image files and fixed shared PINHOLE intrinsics; mismatched
per-image K is rejected explicitly. It runs on CPU using pycolmap 4.2.1 and exports
the largest reconstructed model, recording the number of models and registration
rate. No dense COLMAP reconstruction is attempted.

Run `notebooks/vggt_colab.ipynb` on a Colab GPU after publishing the current source
or copying the adapter into Colab. It uploads your image ZIP and invokes
`code/baselines/vggt_infer.py`. The adapter has not yet been run on a GPU here.
Locally, convert and evaluate downloaded output:

```bash
python code/baselines/convert_to_common.py --vggt data/my_scene/vggt_out.npz --out outputs/my_scene_vggt
python code/eval/run_eval.py --estimate outputs/my_scene_vggt --reference data/my_scene/gt_poses.json --reference-kind gt --out results/my_scene/vggt.json
```

For self-captured scenes use COLMAP's `poses.json` with `--reference-kind colmap`;
these metrics measure agreement with a pseudo-reference. VGGT forward-pass timing
excludes model loading/preprocessing; it is not directly equivalent to end-to-end
CPU pipeline runtime. Do not compare its resized-image intrinsics to original
resolution pixels.

```bash
python code/run_experiments.py --scene sample_data/scene_small --out results/ba --ablation ba
python code/run_experiments.py --scene sample_data/scene_small --out results/views --ablation views --views 5 8 --seed 0
python code/run_experiments.py --scene sample_data/scene_small --out results/detector --ablation detector
python code/eval/make_tables.py --results results --out report/tables
python code/viz/render_orbit.py --reconstruction outputs/sample_ours --out outputs/sample_orbit.mp4 --video
```

The CLI supports `--detector sift|rootsift|orb`, `--ba none|local|local_global`,
`--no-ba`, and `--max-images`. Ablation runs log failures rather than inventing
results. View subsets are shared by filename and evaluated against their matching
GT subset. Different view counts exceeding the available image count are skipped.
`configs/sfm_low_texture.yaml` and `configs/sfm_relaxed.yaml` are explicit alternate
settings; they are not guaranteed to improve a scene.

## Conventions and limitations

All exported poses use OpenCV world-to-camera `x_cam = R @ X_world + t`;
camera center is `C = -R.T @ t`. Reconstructions have arbitrary global scale.
Evaluation aligns camera centers by Umeyama Sim(3). ATE uses shared registered
images; pose AUC counts unregistered pairs as failures. Co-located reference
cameras have undefined translation direction and are excluded from pose AUC,
with the excluded count reported. At least three non-collinear shared cameras
are required for alignment. Clouds are density matched when cloud metrics are
explicitly requested. Reprojection errors refer to processed input pixels.

Bundle adjustment fixes the initial world's first pose and constrains the initial
baseline length; the second camera's orientation/direction can refine. Local BA
optimizes connected cameras while fixing cameras outside the window. The pipeline
retries failed registration candidates after later successes, and filters points
by depth, parallax, and reprojection error. Features are cached by image content,
settings, and OpenCV version. There is no loop closure, multicomponent merging,
dense reconstruction, or mesh reconstruction. Sparse-ring real scenes can still
produce incomplete models even with low residual errors.

## Files and measured status

- `code/sfm/`: geometric core and optimization.
- `code/data/`: calibrated Middlebury download and original sample generation.
- `code/baselines/`: CPU COLMAP and GPU VGGT adapters/conversion.
- `code/eval/`: alignment, metrics, evaluation, CSV/LaTeX generation.
- `code/viz/`: headless PNG/orbit rendering.
- `tests/`: synthetic mathematical tests and image-based reconstruction tests.
- `validation/`: retained measured summaries; large raw outputs remain local.
- `NOTES.md`: validation commands, results, and remaining work.

The sample can be regenerated with `python code/data/make_sample.py`. The project
still needs broader real-scene validation, a VGGT GPU run, the complete mandatory
experiment matrix, the final IEEE report, the narrated demo, and submission
packaging. `AGENT_PIPELINE.md` is the original plan; it is not a completed-work log.

## Third-party software and team

We use library APIs; external implementations are not copied into the geometric
core. Credit [OpenCV](https://opencv.org/), [SciPy](https://scipy.org/),
[COLMAP](https://github.com/colmap/colmap), and
[VGGT](https://github.com/facebookresearch/vggt). The VGGT code/checkpoint terms
are documented in its [license](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt).
No weights are included. Actual Python package versions accompany run summaries.

Repository contributors: Shouvik103 and DagarMihir. Five technical workstreams can
be shared among up to four course group members; they do not imply five members.
Record real member details and contributions in `TEAM_CONTRIBUTIONS.md` before
submission. Git author metadata alone is not evidence of implementation ownership.
