# Reconstruction Lab

Reconstruct 3D objects from overlapping photographs and compare **Our SfM** with
**CPU COLMAP**. The project produces camera poses, colored point clouds and
optional triangle meshes, with a browser interface for viewing results and
exploring experiments.

## Pipelines

```text
Photos → image preparation and calibration
       → Our SfM or CPU COLMAP
       → camera poses and sparse colored points
       → SGBM stereo or optional monocular depth
       → depth filtering → TSDF fusion → triangle mesh
       → interactive viewer and PLY / OBJ / GLB downloads
```

A separate video command automatically selects frames and runs Our SfM to
produce a sparse reconstruction.

## Implementation

- **Our SfM** implements normalized eight-point estimation, RANSAC, essential-matrix
  pose recovery, triangulation, feature tracks and sparse bundle adjustment.
- **OpenCV** supplies SIFT/ORB features, descriptor matching, homography checks,
  PnP and SGBM. **SciPy** supplies nonlinear optimization.
- **COLMAP**, through pycolmap, independently estimates cameras and sparse geometry
  on CPU. Compare both runs the two methods on identical uploaded photos.
- **Meshing** uses Open3D TSDF fusion and cleanup. Optional Depth Anything V2 Small
  predictions are aligned to each method's sparse geometry before fusion.
- **Interface** uses a local Python server, Three.js geometry viewing and Anime.js
  animation. Photos and run history are stored locally in `outputs/web_ui/`.

## Install

Use **Python 3.11**. Run these commands from the repository root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-colmap.txt
```

This installs the pinned project dependencies, including the COLMAP baseline.
Camera estimation and SGBM run on CPU. Optional monocular depth uses MPS on
supported Macs or CPU and downloads its model on first use.

## Run the interface

```bash
python code/web/server.py
```

Open [localhost:8765](http://127.0.0.1:8765).

1. Choose Our SfM, COLMAP or Compare both.
2. Upload overlapping photos and select depth, matching resolution and mesh quality.
3. Click **Reconstruct in 3D**.
4. Inspect the point cloud or mesh, open an original photo with **Reference**,
   and download the outputs.

Uploads accept **3–200 photos**, **25 MB per photo** and **1 GB per run**.
Start with about 60 sharp views covering the object from several heights.
Object quality, SGBM and Balanced mesh are the default settings.

The four tabs provide uploads and saved runs, pipeline explanations, performance
comparisons, and experiment evidence. Your comparisons lists jobs created with
Compare both. Independently saved runs appear in Workspace.

To use another port or a separate run directory, run the following command.

```bash
python code/web/server.py --port 8766 --data-dir outputs/web_ui_preview
```

Stop the server with **Ctrl+C**.

## Run the included samples

The eight-photo synthetic sample includes known camera calibration and poses.
It runs without downloading a dataset or model weights.

```bash
python code/run_sfm.py --images sample_data/scene_small/images --out outputs/sample_ours --config configs/sfm_default.yaml
python code/eval/run_eval.py --estimate outputs/sample_ours --reference sample_data/scene_small/gt_poses.json --reference-kind gt --out results/sample/ours.json
python code/viz/render_orbit.py --reconstruction outputs/sample_ours --out outputs/sample_ours/preview.png
```

An offline analytic sphere sample exercises mesh fusion and export.

```bash
python code/mesh/run_mesh.py --scene sample_data/mesh_sphere --out outputs/sphere_mesh --config configs/mesh_sample.yaml
python code/viz/view_mesh.py --mesh outputs/sphere_mesh/mesh_light.glb
```

For a local video, use a fresh output directory.

```bash
python code/run_video.py --video path/to/video.mp4 --out outputs/video_run
```

## Experiments and tests

The interface presents **nine studies** covering method comparison, bundle
adjustment, photo count, feature detectors, calibration, matching sensitivity,
failure cases and mesh evaluation. The active collection contains **52 rows**
derived from retained development measurements.

Temple has ground-truth camera poses. Car and fountain camera scores measure
agreement with COLMAP references. Mesh studies use a fountain reference scan.
Partial registration, failed runs and incomplete surfaces are reported alongside
successful results. Higher resolution and point counts alone do not establish
better geometry.

```bash
python -m pytest -q
```

Detailed protocols and results are in [EXPERIMENTS.md](docs/EXPERIMENTS.md) and
[EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md). The public
[GitHub Pages archive](https://shouvik103.github.io/3d-cv-project/) displays saved
results and experiments. Live reconstruction runs through the local server.

## Source layout and credits

| Directory | Purpose |
| --- | --- |
| `code/sfm/` | Our camera reconstruction and bundle adjustment |
| `code/baselines/` | COLMAP execution and common-format export |
| `code/mesh/` | Depth, filtering, fusion and mesh evaluation |
| `code/web/` | Local API, interface and static gallery builder |
| `code/experiments/`, `code/eval/` | Experiment execution and scoring |
| `configs/`, `tests/`, `sample_data/` | Settings, regression tests and small inputs |

Our implementation and integration code wrap the third-party libraries above.
COLMAP's reconstruction internals and the pretrained depth model are supplied by
those projects. Meshes use vertex colors. Real handheld-video validation and the
final course report, demo and submission remain separate deliverables.

Third-party sources include [COLMAP](https://github.com/colmap/colmap),
[OpenCV](https://opencv.org/), [SciPy](https://scipy.org/),
[Open3D](https://www.open3d.org/),
[Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf),
[Three.js](https://threejs.org/) and [Anime.js](https://animejs.com/).

See the [active project plan](docs/ACTIVE_PIPELINE.md),
[interface details](docs/EXTENDED_INTERFACE.md) and
[demo recording script](docs/DEMO_SCRIPT.md) for further information.
