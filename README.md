# Incremental Structure from Motion: Our SfM vs. COLMAP

Incremental sparse Structure from Motion from overlapping images, compared with
CPU COLMAP. The geometric implementation includes
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

Dependencies are pinned to the versions used for local validation. Camera estimation
and SGBM run on CPU. Torch/Transformers remain for optional Depth Anything V2 Small
monocular depth. The sparse demo needs no GPU, model weights or dataset download.

## Local upload interface

Launch the browser interface from the repository root:

```bash
.venv/bin/python code/web/server.py
```

Open [the local interface](http://127.0.0.1:8765). Choose or drag in JPEG, PNG or TIFF photos,
enter a project name, then click **Reconstruct in 3D**. The page shows upload
thumbnails, reconstruction stages, registration counts, an interactive point
cloud and triangle mesh, and PLY / OBJ / GLB downloads. Drag to orbit, scroll
to zoom, right-drag to pan, and hover for XYZ coordinates. Point size can be
adjusted from 0.3× to 3× and is remembered across views and reloads. Mesh appearance
can switch between photo colors, shading, wireframe and normals.

The extended interface adds four keyboard-accessible tabs:

- **Workspace** keeps uploads, job progress, saved reconstructions, deletion,
  downloads and the 3D viewers. Selected-run details include measured residuals,
  timing, mesh topology, reconstruction growth, configuration and versions.
- **Pipeline & methods** explains Our SfM and CPU COLMAP, including camera
  calibration, depth/fusion, model provenance and limitations.
- **Performance** provides scene/method filters, interactive bar/line plots,
  sortable measurements, ablations, failures and mesh-quality comparisons.
- **Experiments** exposes the original E1–E7/M4/M5 PNG/PDF plots, CSV/JSON tables,
  reports and embedded interactive benchmark meshes.

Research views load measured, versioned evidence from the latest
`validation/experiments_*/` collection, without needing raw datasets or model
weights. Undefined metrics remain unavailable and failed attempts remain visible.
The benchmark tabs are saved experiments. For new uploads, choose **Our SfM**,
**COLMAP**, or **Compare both**. Compare both creates two independent, serially
queued method runs from identical images; each appears in Your reconstructions.
Open **Performance → Your comparisons** for their measured tables, graphs,
per-method previews, 3D links and CSV/JSON/PNG downloads.

Our SfM and COLMAP offer SGBM stereo or pinned Depth Anything V2 Small with
sparse-depth alignment, then TSDF. SGBM is the upload default and uses calibrated
view pairs. Monocular depth is useful when stereo coverage is difficult, but its
learned prior does not guarantee correct geometry. Classical methods share an
image-only mono-depth cache within a comparison. COLMAP camera estimation runs
in a separate process from mesh fusion. Both methods accept 3–200 photos; about
60 sharp, overlapping views covering every side are recommended.

Object quality is the new upload default: supplied alpha or a confident
simple-background mask excludes background features and depth. Matching and
depth inputs crop to a padded foreground box with exactly adjusted pixel
calibration, spending the resolution budget on the object rather than empty
background. Our SfM retains
valid tracks when conflicting edges occur, checks usable initialization tracks,
and bounds pair matching and BA work. COLMAP masks features and refines focal
length only when input K is estimated. Balanced mesh quality (640px / grid192)
is the default to limit memory. Choose Standard for full-scene captures.

Create new comparisons directly from object folders in `datasets/`:

```bash
.venv/bin/python code/web/seed_datasets.py --dry-run
.venv/bin/python code/web/seed_datasets.py --url http://127.0.0.1:8766
```

The importer takes up to 60 sharp photos, one per natural capture-order bin,
to retain coverage. Temple and Dino have 16 photos each and use all 16 with
their supplied intrinsics. GT poses are never passed to reconstruction.
Raw datasets and outputs are ignored by Git. The importer uses the normal
upload API, so every method appears in Workspace and Your comparisons.

Known original-resolution fx/fy/cx/cy and Our SfM's matching ratio can be entered
under Camera & matching settings. Maximum quality retains full camera frames,
uses up to 12,000 features and exhaustive matching, and spends more CPU time on
camera refinement. Photo matching size includes a 2400px option; source images
are never enlarged. Calibration sources and per-view diagnostics are downloadable.
Finished runs can be retried with explicit depth/matching/quality settings.
Choose **Reuse saved cameras · rebuild mesh only** to change mesh quality or
depth without rerunning camera estimation. Camera and matching changes require a full
retry. Previous attempts and logs are retained locally instead of overwritten.

New Object-quality classical point clouds display up to 115,000 points sampled
from accepted foreground / consistency-filtered depth estimates, capped at one
million exported samples. The smaller TSDF surface cloud remains a separate
download. Neither cap creates points or completes unseen surfaces. Point count
is not a reconstruction-quality score. Camera + mesh timing combines the original camera solve
with the current mesh stage; last-attempt timings and reuse flags are separate.

The active research evidence is `validation/experiments_20261119_classical`.
It projects the retained classical measurements into two-method tables and plots;
this does not claim new benchmark runs. Browser checks and regression evidence
are in `validation/web_classical`. Uploaded photos and full geometry remain local
under the chosen output data directory. Earlier mixed-method records are archives.
No new chart-library or Python dependencies are needed.

If another interface is already using port 8765, launch a separate preview:

```bash
.venv/bin/python code/web/server.py --port 8766 --data-dir outputs/web_ui_extended
```

Open [the extended preview](http://127.0.0.1:8766). This separate data directory
keeps preview uploads apart from your original run history. Omit `--data-dir`
when you want to use the existing `outputs/web_ui` library.

Use **Dark mode** in the header to switch themes. The interface follows the
system appearance until you choose a theme, then remembers that choice locally
across reloads. On narrow screens the toggle appears as a moon icon. Uploads,
geometry colors and reconstruction settings are unaffected by theme changes.

Use 12–60 sharp, overlapping views around a stationary object for a first run;
the interface accepts 3–200 images, up to 25 MB per image and 1 GB per run.
All uploaded images are used as reconstruction candidates. Photo matching size
controls the images used to estimate camera poses. Camera settings optionally accept a known focal length
in pixels at the original image resolution; otherwise the existing EXIF /
estimated-intrinsics preparation is used. Keep the same camera and lens.

**Mesh quality** is independent of photo matching size:

| Preset | Maximum depth-map longest edge | TSDF grid setting |
| --- | ---: | ---: |
| Balanced | 640 px | extent / 192 |
| High (new UI default) | 960 px | extent / 256 |
| Ultra | 1280 px | extent / 384 |

Higher presets estimate depth from more pixels and use smaller surface voxels;
they take more CPU time and memory. Mesh images are prepared separately from
original uploads, with camera intrinsics and tracked pixel observations scaled
to the new image size. A smaller photo matching size does not cap mesh resolution.
Photos smaller than the selected depth size are not enlarged. Original uploads
and the sparse reconstruction are unchanged. Older jobs without a stored quality
continue using Balanced. The choice applies to new runs; existing car/chair
results are not regenerated automatically.

The interactive light mesh remains capped at 80,000 triangles for responsiveness.
Use **Full mesh PLY** to download the full extracted mesh. Higher settings can
preserve finer observed detail, but cannot recover unseen surfaces or fix blurry,
reflective or poorly overlapping photos.

Reconstruction uses the existing SfM and CPU stereo / TSDF mesh pipeline, with
all registered views eligible for fusion. No model weights are downloaded.
The point cloud stays available when mesh generation fails, with the reason
shown in the page. Partial registration, failed reconstruction and cancellation
are shown explicitly. Queued runs execute one at a time; the page can be closed
while the server continues processing. Saved runs can be reopened from the
reconstruction library. Stopping the server interrupts processing.

Every library card has a **Delete** button with confirmation. Deleting an active
run stops its worker; deleting a queued run prevents it from starting. Uploaded
runs and their outputs move to `outputs/web_ui/.trash/`. Deleting a saved car /
chair entry removes it from the library while preserving its original source
files. The removal persists across page reloads and server restarts.

Active runs show an estimated percentage and progress bar in both their library
card and workspace. Feature extraction, pair matching, camera registration and
depth estimation report completed work counts. The overall percentage weights
these stages; it is not an elapsed-time estimate. Long optimization or fusion
steps can stay at one percentage while their work finishes. Only a finished run
reaches 100%.

Photos, logs and outputs stay under `outputs/web_ui/<run-id>/`. The server binds
only to this computer's loopback address. The browser loads Three.js from
jsDelivr and fonts from Google Fonts; internet access is needed for the 3D
viewer, while images are sent only to the local server. Existing car and chair
results appear as examples when their local `results/mesh/` files are present.
Those original datasets/results are not bundled in the repository.

The point preview shows up to 115,000 points and labels the displayed and full
counts separately; downloads retain the full cloud. The saved car and chair use
their original foreground clouds (2,232,430 and 3,186,516 points respectively),
each previewed with exactly 115,000 sampled points. The mesh viewer shows all
triangles from the exported light mesh. Coordinates use a normalized upright
display frame in arbitrary units; exported geometry keeps its native frame.
Unobserved surfaces and holes are not filled artificially.

Use `--port 8766` for another port, or `--data-dir /path/to/local/runs` to choose
the run directory. Stop with Ctrl+C. Existing image/video CLI commands remain
available.

## Command-line sample

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

## Reconstruct from a video

Provide a local video file; frame timestamps are chosen automatically:

```bash
python code/run_video.py --video "path/to/object.mp4" --out outputs/my_video
```

Keep the object stationary and walk the camera slowly around it. Keep the same
lens/zoom, maintain focus, and use a textured subject in good light. Panning from
one stationary position or rotating the object against a fixed background does
not provide the assumptions this pipeline needs. The output is a **sparse colored
point cloud and camera poses**, not a textured mesh or a solid model.

Frame selection scans a bounded set of candidates across the video, rejects blur
and weak texture, separates shots when feature tracking loses overlap, and chooses
the continuous shot with the most usable image motion. Forward/backward optical
flow consistency estimates overlap; motion between retained views determines
spacing. The motion threshold adapts for short clips. Redundant views are skipped,
and at most 40 frames enter the default reconstruction. If that budget cannot
preserve overlap, the command asks for a shorter clip or a larger budget instead
of silently connecting unrelated views. Image motion is a proxy for viewpoint
change; the SfM initialization still checks actual triangulation/parallax.
The cloud can include the static background; this workflow does not segment the
object automatically.

Outputs under the selected `--out` directory:

- `input/images/`: the selected frames.
- `input/frames.json`: original frame indices, timestamps, quality scores,
  tracking overlap, candidate decisions, video hash, and selection settings.
- `input/selected_frames.png`: a contact sheet labelled with timestamps.
- `input/intrinsics.yaml`: camera intrinsics scaled to the extracted image size.
- `reconstruction/points.ply`, `poses.json`, and `summary.json`: the reconstruction.
- `preview.png`, `run.log`, and `status.json`: preview, progress, and final state.

Use a fresh output directory for every run. `status.json` distinguishes completed,
partial, failed, and extraction-only runs. Partial registration is reported and
returns exit code 2; it is not labelled a complete reconstruction. Unsupported,
static, blurry, or disconnected video fails with an explanation, preserving
diagnostics. MP4/MOV/AVI support depends on the locally installed video codecs.
Decoder timestamps are retained where available, with a documented frame/FPS
fallback. Orientation metadata is handled by supported OpenCV backends.

The default focal length is an approximation because ordinary decoded video has
no reliable EXIF focal length. Supply calibration or a known focal length for
better geometry; both refer to the **original decoded video resolution**:

```bash
python code/run_video.py --video object.mov --out outputs/calibrated_video --calibration data/calibration.yaml
python code/run_video.py --video object.mp4 --out outputs/known_focal --focal-px 1200
python code/run_video.py --video object.mp4 --out outputs/frame_check --extract-only
python code/run_video.py --video object.mp4 --out outputs/more_views --max-frames 60
```

Optional `--video-config configs/video_default.yaml` controls selection;
`--config configs/sfm_relaxed.yaml` changes reconstruction settings. More frames
increase exhaustive-matching cost. No timestamps need to be entered manually.

To try the complete workflow without supplying your own footage:

```bash
python code/data/make_sample_video.py --out outputs/video-example.avi
python code/run_video.py --video outputs/video-example.avi --out outputs/video-example --focal-px 700
```

This fixture contains the generated multiview scene, unequal pauses, duplicate
views, blur, and a cut to an unrelated shot. It is a controlled integration test;
real handheld video has not yet been validated. The selector uses OpenCV's
[optical flow](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)
and [video decoding properties](https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html).

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

The active comparison uses Our SfM and CPU COLMAP only. See
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for E1–E7 and M4/M5 commands.
For self-captured scenes use COLMAP's `poses.json` with `--reference-kind colmap`;
these metrics measure agreement with a pseudo-reference. Workers record end-to-end
method time, and mesh-only retries label the reused camera solve separately.

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
settings, and OpenCV version. The sparse stage has no loop closure or multicomponent merging. Dense depth
and mesh reconstruction are available through the separate mesh add-on below. Sparse-ring real scenes can still
produce incomplete models even with low residual errors.

## Files and measured status

- `code/sfm/`: geometric core and optimization.
- `code/data/`: calibrated Middlebury download and original sample generation.
- `code/baselines/`: CPU COLMAP baseline and common-format conversion.
- `code/eval/`: alignment, metrics, evaluation, CSV/LaTeX generation.
- `code/viz/`: headless PNG/orbit rendering.
- `tests/`: synthetic mathematical tests and image-based reconstruction tests.
- `validation/`: retained measured summaries; large raw outputs remain local.
- `NOTES.md`: validation commands, results, and remaining work.

The sample can be regenerated with `python code/data/make_sample.py`. Sequential
E1–E7 and mesh M4/M5 runs, measured outcomes and limitations are documented in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) and
[docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md). E5 uses calibrated Temple
as approved by the user; real car checkerboard calibration remains unavailable.
The final IEEE report, narrated demo, contributor evidence and fresh submission
packaging remain separate work. `AGENT_PIPELINE.md` is the active two-method plan, not a completed-work log.

## Third-party software and team

We use library APIs; external implementations are not copied into the geometric
core. Credit [OpenCV](https://opencv.org/), [SciPy](https://scipy.org/),
[COLMAP](https://github.com/colmap/colmap), [Open3D](https://www.open3d.org/),
and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
No weights are included. Actual Python package versions accompany run summaries.

Repository contributors: Shouvik Das (23bcs125; Shouvik103) and
Mihir Dagar (23bcs080; DagarMihir). Five technical workstreams can
be shared among up to four course group members; they do not imply five members.
Record real member details and contributions in `TEAM_CONTRIBUTIONS.md` before
submission. Git author metadata alone is not evidence of implementation ownership.

## Mesh and mesh view

The optional mesh stage preserves the existing SfM pipeline. It loads the exact
resized input images, common poses, sparse points and observations, verifies the
OpenCV world-to-camera convention, predicts or computes depth, and fuses a
vertex-colored triangle surface. The ordinary photo and video commands still
produce their existing point-cloud outputs; meshing is a separate command.

```bash
# Offline: own SfM outputs and cached Depth Anything predictions ship with this sample.
python code/mesh/run_mesh.py --scene sample_data/scene_small --out outputs/mesh --config configs/mesh_default.yaml --cache-only
python code/viz/view_mesh.py --mesh outputs/mesh/mesh_light.ply
python code/viz/view_mesh.py --mesh sample_data/scene_small/mesh_light.glb --mode shaded
python code/viz/view_mesh.py --mesh outputs/mesh/mesh_light.ply --screenshot outputs/mesh/view.png
python code/viz/render_mesh.py --mesh outputs/mesh/mesh_light.ply --out outputs/mesh_renders --orbit
# Analytic offline fixture, explicitly not a learned-depth prediction:
python code/mesh/run_mesh.py --scene sample_data/mesh_sphere --out outputs/sphere_mesh --config configs/mesh_sample.yaml
```

New scenes need `poses.json`, `points.ply`, input images, and ideally
`observations.json`. A `scene.json` makes paths explicit and portable:

```json
{"reconstruction":"reconstruction","images":"images","depth_cache":"depth_mono","masks":"masks"}
```

`masks` is optional; its per-image filenames are `image.jpg.png`. Transparent PNG
alpha is used automatically when no separate masks are specified. Paths in this
manifest are relative to its folder. Alternatively pass `--scene outputs/my_scene
--images data/my_scene/images`. Supported existing exports include our own SfM,
common COLMAP exports (with radial undistortion on load). An unregistered image
never enters depth generation or fusion.

For new learned-depth predictions, the first run downloads the verified
[Depth Anything V2 Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf),
Apache-2.0, pinned at revision `5426e4f0f36572d16453bbda7a8389317b1bef99`.
The model runs sequentially on MPS when available, otherwise CPU. Predictions are
relative **inverse depth**, not meters. RANSAC plus least squares fits
`1/z_sfm = s*d_rel + b`, requiring 30 inliers and median relative residual <10%.
Depth caches include the image content, model/settings key and model provenance;
weights stay in the external Hugging Face cache and are never packaged.
`configs/mesh_mono_small.yaml` lowers inference to 384px for smaller scenes.

```bash
# Classical stereo fallback; no depth-model weights or GPU needed.
python code/mesh/run_mesh.py --scene my_scene --out outputs/mesh_sgbm --config configs/mesh_sgbm.yaml
# Thin-object ROI alternative; a wider box may also retain more background.
python code/mesh/run_mesh.py --scene my_scene --out outputs/mesh_object --config configs/mesh_object.yaml
```

TSDF uses Open3D **0.19.0**: 0.20.0's scalable TSDF returned empty geometry even
for a simple analytic plane on the development Mac. The version is pinned after
that regression was isolated. Unit tests cover sphere/box depth, pose inversion,
robust depth alignment, SGBM metric depth, large-image consistency, mesh cleanup,
color-preserving PLY/OBJ/GLB round trips and window-free rendering. OBJ stores
colors through the common per-vertex RGB extension; GLB/PLY are preferable for
portable vertex colors. Meshes also open in MeshLab or Blender.

Outputs include full and ~80k-triangle light meshes, a fused point cloud,
`alignment.json` with all accepted/skipped images, `convention.json`, a projection
overlay, `config.yaml`, `summary.json` with geometry/timing/memory and held-out
metrics, and a preview. `status.json` records CLI failure/completion. Every frame
uses OpenCV world-to-camera extrinsics. Multi-view consistency handles occlusion,
and ROI, gradients, component thresholds and voxel sizes are configurable.

The default three held-out registered images are excluded from depth generation,
stereo neighbors, consistency checks and fusion. Their photos are evaluated only
where the rendered surface is visible, with PSNR/SSIM/MAE **and coverage**. This
is a held-out **fusion** test: the poses and sparse structure still come from SfM
on all images. It is not an entirely held-out SfM/training experiment. Low
coverage and non-watertight meshes remain explicit limitations. Monocular depth
can smooth or thicken surfaces; glossy, textureless and thin regions can have
holes. Do not interpret a point cloud or a Poisson-completed surface as observed,
watertight ground truth.

```bash
python code/mesh/run_experiments.py --scene my_scene --out results/mesh_ablations --include-stereo
python code/mesh/failure_study.py --out results/mesh_failures
python code/mesh/mesh_eval.py --estimate outputs/a/mesh_full.ply --reference outputs/b/mesh_full.ply --estimate-scene scene_a --reference-scene scene_b --kind agreement --out results/mesh_agreement.json
python code/mesh/package_submission.py --report report.pdf --demo demo.mp4 --out dist/submission.zip
```

Geometry comparison aligns shared camera centers with the existing Umeyama
Sim(3). `--kind gt` crops the estimate to the reference mesh bounds and reports
accuracy/completeness/F-scores at 0.5%, 1% and 2% of its diagonal. `--same-frame`
is required to explicitly bypass camera alignment. Cross-method agreement is not
ground-truth accuracy. No real GT triangle mesh was supplied for these local
runs; no GT-mesh result is claimed. Middlebury's supplied camera calibration
supports pose evaluation, not mesh ground truth.

Render output includes three colored/shaded/wireframe/normal stills, depth maps,
and optional 150-240-frame colored and shaded orbits. CPU `RaycastingScene`
requires no display server or OpenGL. For upright display, add `--poses
path/to/poses.json`; `--up x|y|z` corrects captures with rolled images. This only
rotates the displayed mesh. `--screenshot` works without opening a window; GUI
controls and optional camera/point overlays remain available in `view_mesh.py`.

Measured run summaries, figures and limitations are mirrored in
`validation/mesh/`; generated full outputs remain under `results/mesh/`.
The report addendum and 30-second mesh demo segment are integration assets;
they do not replace the final full IEEE report, narrated 1-2 minute demo or human
portal submission. Packaging excludes raw datasets, weights, virtual
environments and feature caches; it enforces <200 MB and <40 MB sample data.

### Third-party code and modifications (mesh add-on)

| Component | Use / modification |
|---|---|
| [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 checkpoint, inference only through Transformers; no training or copied model implementation. |
| [Open3D](https://www.open3d.org/docs/0.19.0/) | TSDF/marching cubes, Poisson, cleanup/decimation, CPU ray casting, native viewer; API calls only. |
| [trimesh](https://trimesh.org/) | Color-preserving OBJ/GLB export and import; library API only. |
| [scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html) | Held-out SSIM; library API only. |
| Our additions | Adapters/convention gate, inverse-depth alignment, filtering/consistency, SGBM depth conversion, fusion orchestration, cleanup/export, evaluation, viewers/rendering, fixtures, experiments and packaging under `code/mesh/` and `code/viz/`. |
| Existing files modified | The mesh stage consumes common SfM/COLMAP exports. Requirements, configs, README/NOTES and sample cache ignore exceptions are updated. |

To regenerate the integration media after rendering the logged scenes, run
`python docs/build_mesh_demo.py`. To rebuild the PDF from mirrored results,
install `requirements-report.txt` and run `python docs/build_mesh_report.py`.
These commands run from the repository root.

### Interactive browser mesh

For a rotatable surface in the browser, export the light mesh as a self-contained
HTML view (no local server or mesh-file fetch needed):

```bash
python code/viz/view_mesh_web.py --mesh outputs/mesh/mesh_light.ply --out outputs/mesh/interactive_mesh.html --title "My interactive mesh"
```

Open the resulting HTML file. Drag to orbit, scroll/pinch to zoom, right-drag to
pan, or choose a preset viewing angle. Appearance modes are colored surface,
shaded, wireframe and normals. Axes can be toggled; hovering reports the actual
triangle surface's X/Y/Z in upright display coordinates, with arbitrary scale.
Optional `--poses poses.json --front-image name.jpg` orients the object for display.
The browser retains **every triangle in the supplied light mesh**; only display
positions are quantized to 16 bits. Original PLY/OBJ/GLB geometry is unchanged.
The embedded view uses gzip decoding and pinned Three.js/OrbitControls 0.128.0
from jsDelivr; an internet connection is needed to load those scripts once.
`--fragment-out` optionally exports an inline view, enforcing a 1 MB limit.

Car and chair exports are retained locally at
`results/mesh/car_sgbm/interactive_mesh.html` and
`results/mesh/chair_sgbm_wide_roi/interactive_mesh.html`. Their sidecar JSON files
record original hashes, exact counts and display transforms. Browser checks
verify mouse orbit/zoom/pan, all appearance modes, axes, surface hover geometry
and mobile layout. The native Open3D viewer remains available.
