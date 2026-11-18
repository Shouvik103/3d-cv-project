# Implementation and validation notes

## Fresh same-upload comparisons

The extended branch now runs Our SfM, COLMAP, VGGT, or All three for uploaded
images. All three creates three independent queued jobs with identical photo
bytes and explicit method labels, plus a live comparison with CSV/JSON/PNG
exports. Classical methods offer SGBM or monocular depth / sparse alignment;
VGGT uses its own cameras/depth in its isolated runtime. COLMAP camera inference
is isolated from learned depth to avoid a native OpenMP threading conflict.

Nine fresh Temple/car/chair jobs were created through the normal HTTP API and
remain in `outputs/web_ui_extended`. All have cloud and mesh outputs. Camera
counts (Our SfM / COLMAP / VGGT) are Temple 16/13/16 of 16; car 4/10/30 of 30;
chair 20/30/30 of 30. Partial models and surface artifacts are retained. Car and
chair input intrinsics were earlier estimates, not GT; no earlier poses or
geometry were used as fresh method outputs. New-upload ATE/AUC is explicitly
COLMAP agreement, not GT accuracy.

Retry controls now have independent per-run fields initialized from saved
settings; unrelated upload-form defaults cannot silently replace them. Prior
attempts, including failed retries, remain archived. Queue generation tokens
prevent a cancelled queued task from launching a later retry twice.

Validation: 68 tests pass, all prepared-photo hashes match across methods within
each group, and browser checks confirm the nine library entries, method choices,
live previews, model opening, comparison tables, and saved retry settings.
Compact tables, metadata, plots, screenshots and nine interactive mesh HTMLs
are retained in `validation/web_three_methods/`. Raw photos and large geometry /
prediction arrays remain local. Untracked user dataset folders are not included.

## Extended interface — 18 November 2026

Branch `codex/extended-reconstruction-interface` adds Workspace, Pipeline &
methods, Performance and Experiments tabs. The research views read the saved
67 conditions from all nine studies; new uploads retain their own pipeline and
do not automatically launch COLMAP/VGGT. The selected-run inspector uses that
run's actual summaries, with mean/median residual distinctions preserved.

The complete suite passes 63 tests. Browser checks cover filtering, missing
metrics, sorting, keyboard navigation, light/dark layouts, original evidence
links and both existing 115,000-point car/chair previews. The 390px layout has no
document-level horizontal overflow. Compact screenshots and validation are in
`validation/web_lab/`; interface details are in `docs/EXTENDED_INTERFACE.md`.
Preview server: port 8766 with `outputs/web_ui_extended`, separate from the
existing port-8765 service and its run history.

## Current status — 18 November 2026

The dated entries below preserve earlier milestones. Statements in the November
14 entries about absent VGGT inference, GT mesh files and the experiment matrix
describe that earlier state, not the current implementation. The sequential
E1–E7/M4/M5 protocol is in `docs/EXPERIMENTS.md`; the measured consolidated record
is `docs/EXPERIMENT_RESULTS.md` and `validation/experiments_20261118/`.
Real VGGT predictions now run on Apple MPS in an isolated compatible runtime.
E5 uses calibrated Temple with the user's approval; no car checkerboard capture
or calibration is claimed. Failed and partial reconstructions remain recorded.

All nine requested stages were executed in order. M4/M5 use the original EPFL
fountain-P11 GT triangle mesh, with camera-only Sim(3), GT-box crop, 200k surface
samples per direction, error colors and three interactive mesh exports. Source
landmarks/ROI and VGGT's different depth backend remain explicit confounds.
The full current test suite passes 59 tests; main/VGGT dependency checks,
compilation and Git whitespace checks pass. A real VGGT no-track mesh path exposed
a NumPy-boolean JSON serialization bug, now fixed and regression-tested; its
failed attempt remains archived. Final measurements include poor and partial
results, not just successful-looking previews. No new commit/push was made for
this experiment task.

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

## 14 November 2026 — automatic video input

Added `code/run_video.py` for local video files, as requested. It selects frames
without manual timestamps, writes intrinsics and a timestamped contact sheet,
and invokes the existing image-based reconstruction. No new dependencies are
required.

The selector samples candidates throughout the clip, bounds analysis storage,
rejects weak texture and blur, tracks corners forward and backward to estimate
overlap, separates shots, and chooses the viable continuous shot with the most
image motion. It adapts the movement threshold for short clips and verifies
overlap again after enforcing the frame budget. Original selected frames are
decoded in a second sequential pass rather than relying on imprecise video
seeks. The report includes source frame indices, decoder timestamps or an
explicit frame/FPS fallback, candidate decisions, and the source video hash.

The generated video fixture holds eight known camera views for unequal times,
adds blurred transition frames, and ends with an unrelated static shot. It is
not a real handheld recording. The complete workflow selected six views at
0.00, 1.25, 3.50, 7.25, 10.50, and 12.00 seconds and reconstructed all six with
2,440 points. It rejected seven sampled blurred candidates and six candidates
from the unrelated shot. The selected frame indices are 0, 15, 42, 87, 126, 144;
they are deliberately not uniformly spaced. Retained evidence is under
`validation/video/`.

The expanded tests cover image selection and video-to-SfM accuracy, blur and
duplicate rejection, scene cuts, scaled intrinsics, stationary footage, missing
files, preservation of existing output, invalid settings, and a frame budget
that cannot preserve overlap. All 26 tests pass. The CLI also runs successfully
from outside the repository using its default configuration paths.

Completed, partial, extracted-only, and failed runs have explicit states in
`status.json`; incomplete camera registration returns exit code 2. Existing
output directories are protected. Video without a known focal length uses the
documented approximate K. A checkerboard calibration or `--focal-px` can override
that estimate. Pure rotation, non-rigid subjects, object turntables against a
fixed background, digital zoom, and poorly overlapping footage remain outside
the reliable assumptions of this classical pipeline. Object/background
segmentation and dense meshing are not implemented. A real phone video still
needs validation; no such footage was supplied in this session.

## Mesh add-on implementation and validation (14 Nov 2026)

Implemented `MESH_ADDON.md` as a separate stage. Existing SfM geometry, photo
input and video input paths are unchanged. New files are under `code/mesh/`
(adapters, learned/stereo depth, alignment, filters, TSDF/Poisson, cleaning,
exports, evaluation, CLI, synthetic fixtures, experiments/failure study and
packaging) and `code/viz/{view_mesh,render_mesh}.py`. New configs document default,
low-memory monocular, classical, wider-object-ROI and offline fixture settings.
The only existing inference code modified is `code/baselines/vggt_infer.py`, which
adds raw depth/depth-confidence fields to its NPZ export; existing fields remain.
README, pinned requirements and sample depth-cache ignore exceptions are updated.
`sample_data/scene_small` now includes our SfM output, five cached learned-depth
predictions for its default fusion split and light PLY/GLB meshes. An analytic
sphere fixture is separately labelled; its caches are not network predictions.

Inspected common JSON/PLY/observations formats before implementation. Our poses
already use OpenCV world-to-camera. Existing Temple and generated-tile SfM
reprojection medians were 0.1639 px and 0.1232 px. Adapters validate image sizes,
registered poses, K, track ordering and radial undistortion, retaining overlays.
No SfM export rewrite was needed. The renderer uses actual unnormalized ray
directions to obtain z-depth from RaycastingScene's t_hit; normalizing without
rescaling t_hit, as in a naive outline, would produce incorrect z-depth.

Tests were written before real mesh runs. Open3D 0.20.0's ScalableTSDFVolume
returned zero points/triangles even for an identity-camera unit-depth plane;
UniformTSDFVolume on the same input returned geometry. After pinning Open3D
0.19.0, the geometric and full pipeline gates passed. Tiny raw TSDF silhouette
components are evaluated after the same component cleanup used in production.
A large-image consistency remap initially exceeded OpenCV's SHRT_MAX dimension
because pixels had been flattened into one long map; using HxW maps fixes it,
and a >32,767-pixel regression test covers it. The synthetic cache fixture uses
OpenCV integer pixel centers and interior landmarks to avoid interpolating
background at sphere silhouettes; its relocated CLI test now checks radial
surface error as well as successful export.

Final whole-project test result: **36 passed** (nine mesh tests). The sole pytest
warning comes from the installed Dash plugin's deprecated hook, not mesh code.
An isolated new virtual environment installed requirements and reconstructed both
shipped samples offline, wrote a headless screenshot and passed the mesh suite.
Exact commands/results are in `validation/mesh/fresh_machine.json`; archive
extraction is recorded in `validation/mesh/archive_check.json` when present.
No hardcoded user-home paths were found under `code/`. `pip check` passed.

Validated package versions: Open3D 0.19.0, torch 2.15.0, torchvision 0.29.2,
transformers 5.23.0, trimesh 5.2.1, scikit-image 0.26.0, tqdm 4.70.1. Depth Anything
V2 Small was verified against its Hugging Face model card (Apache-2.0), and its
revision is pinned to `5426e4f0f36572d16453bbda7a8389317b1bef99`. Actual inference
succeeded on MPS. Weights stay outside the repository. Cached predictions record
model/image/settings provenance; no model download is needed for shipped samples.

Measured runs under `results/mesh/` include:
- `temple_mono`: our 16-view SfM, 13 fused views after holding out 3, 173,812 full
  triangles. This public real scene has no supplied GT triangle mesh.
- `sample_mono` / `sample_sgbm`: our generated tile-scene SfM, five fusion views
  after holding out three; both depth backends produced surfaces.
- `chair_sgbm` and `chair_sgbm_wide_roi`: existing COLMAP poses, 57 fusion views
  after holding out three. The wider ROI retains more of the thin legs. The wide
  run has 100,870 full triangles and 79,999 light triangles.
- `car_sgbm`: existing COLMAP poses, 57 fusion views; 119,296 full triangles.
  Glossy/textureless surfaces and background artifacts remain visible.
- `sphere_cached`: analytic synthetic control; not evidence of learned depth.

`validation/mesh/mesh_metrics.csv` and per-run JSONs contain actual statistics,
PSNR/SSIM/MAE/coverage, runtime and peak RSS. Held-out checks exclude photos from
all depth/fusion neighbors, but SfM poses/landmarks were estimated with all views;
these are fusion checks, not fully held-out SfM experiments. Every mesh remains
non-watertight; no gaps or unseen surfaces were invented. Low image coverage is
reported alongside image scores. No real GT mesh F-score or VGGT mesh result is
claimed: the necessary GT geometry/bounding volume and VGGT depth were absent.
The geometry-comparison CLI supports camera-aligned GT/agreement evaluation when
those files become available.

The tile-scene ablation matrix executed mono/SGBM, TSDF/Poisson, voxel divisions
128/256/384 and consistency on/off. A separately labelled analytic matrix and
controlled wrong-depth-scale/two-view failure study are retained. Headless
renders include three views each of colored, shaded, wireframe and normals,
metric depth arrays and colored/shaded 150-180-frame 1280x720 orbit videos.
Native display orientation follows capture conventions; Temple images are
rolled, so its render command includes `--up y`. This is a display rotation only.

`docs/mesh_report.md`, its generated PDF addendum, measured figures and a
30-second captioned `results/mesh/mesh_segment.mp4` are ready for integration.
Bibliographic entries were verified against original paper/model/library pages.
The report addendum is not the complete IEEE course report, and the captioned
segment is not a narrated 1-2 minute full demo. A bounded review ZIP is generated
with the packaging CLI. Final course report/demo assembly, any absent GT/VGGT
experiments, contributor evidence, and human portal tagging/upload remain
separate project work; no submission or remote push was performed by this task.

## Browser mesh interfaces (14 Nov 2026)

Added `code/viz/view_mesh_web.py` and two literal templates under
`code/viz/assets/`. The exporter embeds the supplied triangle surface with gzip,
16-bit display coordinates, original uint8 vertex colors and exact connectivity.
No further mesh decimation or point sampling is performed. It creates a standalone
browser file and optionally an inline fragment. Sources retain their original
file hashes. Three.js and OrbitControls 0.128.0 provide rendering/orbit controls;
no external implementation was copied into the reconstruction pipeline.

Interactive car: all 80,000 light-mesh triangles, 41,977 vertices; chair: all
79,999 light-mesh triangles, 41,231 vertices. Both interfaces support mouse orbit,
wheel/pinch zoom, right-drag pan, preset views, colored/shaded/wireframe/normals,
axes and ray-cast surface XYZ hover in arbitrary display units. Shared camera
poses provide upright display orientation. The reconstruction, source meshes
and existing point-cloud views are unchanged.

Actual browser validation exercised modes, orbit/zoom/pan, axes and a 360px mobile
layout. Hover XYZ was checked against the selected triangle's plane, within
0.0002 display units. Both fragments are below 1 MB and include every triangle.
Validation JSONs and screenshots accompany the exports under `results/mesh/`.
# Local image-upload workbench (2026-11-15)

Added `code/web/server.py`: a loopback-only browser application that queues
image reconstructions in separate processes and saves jobs under
`outputs/web_ui/`. Launch with `.venv/bin/python code/web/server.py` and open
`http://127.0.0.1:8765`. It supports upload/drop previews, project names,
resolution and optional focal settings, progress, cancellation, persistent run
history, run logs, source thumbnails and downloadable geometry. Existing image
and video CLIs are unchanged. No additional Python dependencies are required.

The worker calls the existing preparation, SfM, SGBM / TSDF and mesh-export code.
Every registered view is eligible for fusion (`heldout.num_views=0` in this
user-facing reconstruction path). Point output survives a failed mesh stage.
The browser switches between point clouds and actual triangle surfaces, with
orbit/zoom/pan, fit, XYZ axes and hover coordinates, and colored / shaded /
wireframe / normal mesh modes. Display coordinates are upright and normalized;
original PLY / OBJ / GLB geometry is untouched. Point previews cap at 115,000 and
state both counts; mesh previews retain every light-mesh face. Three.js and web
fonts load externally; image bytes go only to the local server.

Validation: 39 project tests passed (one existing external Dash deprecation
warning). Three new tests cover upload validation and cross-origin/path
boundaries, restart state, and geometry export fidelity / point preview limits.
An actual Chrome upload of the eight included tile views, with original focal
length 700 px, registered 8/8 cameras and produced 3,116 sparse points,
40,957 dense points and a 74,950-face mesh. Both geometry views, XYZ hover,
all seven download links and a 390 px mobile layout passed browser checks with
no JavaScript exceptions. Existing 80,000-face car and 79,999-face chair
examples loaded. Actual blank-image failure and running-job cancellation were
verified separately. Browser screenshots / run JSON are retained locally under
`outputs/web_ui/qa/`. The sample upload was subsequently removed from the UI
library at the user's request, with its validation artifacts retained in QA.

Corrected the existing car / chair cloud previews to read their original
foreground PLYs rather than TSDF-extracted mesh clouds. Cache names now include
source identity and size / modification time so earlier smaller cached clouds
cannot be reused. Chrome verified exactly 115,000 displayed vertices from
2,232,430 car points and 3,186,516 chair points, plus XYZ hover, full-cloud
download links, a 0.3×–3× point-size slider, persistence across geometry switches
and page reloads, and no overflow at 390px. All three web tests passed; browser
evidence is in `outputs/web_ui/qa/cloud_fix_validation.json` and adjacent PNGs.

Added per-card deletion with a confirmation dialog, including original car /
chair entries and every uploaded run state. Uploaded runs move into local
`.trash/`; active workers stop before their files move, and deleted queued jobs
are skipped by the executor. Original car/chair source files are retained while
their library entries are hidden persistently. Selection clears after deletion.

Added optional work-counter callbacks to SfM and mesh reconstruction, without
changing the geometry algorithms. The worker writes throttled progress for
feature images, every attempted image pair, camera registration, depth views,
fusion, cleanup and export. The browser shows estimated overall percentages and
bars in the workspace and cards; percentages represent stage-weighted work,
rather than an elapsed-time estimate. Logs provide conservative progress for
runs started before callbacks existed. Dense-cloud completion refreshes an
already-open sparse preview instead of retaining the earlier browser geometry.

Isolated Chrome QA verified progress increasing from 0 to 100 through an actual
successful image-to-mesh run, confirmation cancellation, uploaded-run deletion,
persistent example removal and mobile layout, with no JavaScript exceptions.
User runs and saved entries were not deleted by QA. Evidence is retained under
`outputs/web_ui/qa/delete_progress_validation.json` and adjacent screenshots.
Final regression suite: 44 passed, with the existing external Dash warning.
This includes actual SGBM mesh progress, complete pair-attempt counters,
running/queued deletion, persistent example removal and source preservation.

Added a separate mesh-quality setting: Balanced uses depth-side 640 and voxel
division 192; High uses 960 / 256; Ultra uses 1280 / 384. The UI defaults new
runs to High, while jobs without the setting retain Balanced behavior. Renamed
the original control to Photo matching size. Mesh preparation now reads original
uploads separately, scales camera K and tracked pixel observations to those
dimensions, and leaves sparse world coordinates / original SfM files unchanged.
Small photos are not enlarged. This prevents Fast photo matching from discarding
pixels needed for higher-quality depth. Full mesh PLY is exposed in downloads;
the interactive light mesh still retains at most 80,000 triangles.

Validation covers persisted / rejected quality choices, restoring 80px originals
from a 40px matching copy with exact calibration / reprojection and unchanged
world points, plus actual 960px and 1280px textured stereo planes with depth
scale error below 3%. A Chrome High-quality upload completed, with its measured
config confirming depth-side 960 and voxel division 256, working outputs,
progress, deletion and responsive layout. That sample's originals are only
800px, so they remain 800px in depth processing; no pixels were invented.
Evidence: `outputs/web_ui/qa/mesh_quality_validation.json` and adjacent PNGs.
Final mesh-quality regression: 50 tests passed, with the existing external Dash warning.

Added dark mode (2026-11-18): a header toggle with accessible pressed state,
keyboard operation and per-browser persistence. A small independent script
applies the stored / system theme before styles render, follows system changes
until a manual choice, and remains usable when local storage is blocked. Dark
colors cover fields, upload panels, saved cards, statuses, progress, downloads,
logs and confirmation dialogs. Mesh / photo colors and backend code are unchanged.

Chrome validation checked system defaults and changes, keyboard toggle, both
themes across reloads, storage restrictions, dialog dismissal without deleting,
and layouts at 390px and 320px. Measured body, muted and input text contrast
exceeded 4.5:1. Screenshots and results are retained in
`outputs/web_ui/qa/theme_validation.json` and adjacent `theme-*.png` files.
