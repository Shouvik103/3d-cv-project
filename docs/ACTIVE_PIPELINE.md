# Project plan: Our incremental SfM vs. CPU COLMAP

Updated scope: 19 November 2026. This is the active plan; earlier designs and
experiment records remain in Git history. The project implements and compares
**two independent camera-reconstruction methods: Our SfM and COLMAP**.

The course deliverable is `GroupX_3dcv_project.zip`, under 200 MB, submitted
before 22 November 2026. Replace X with the actual group number. Include source,
pinned dependencies, small runnable inputs, a 4–6 page IEEE report and a playable
1–2 minute demo. Actual member details, contributions and portal submission are
human responsibilities. Do not claim these deliverables are complete merely
because the reconstruction code works.

## Objective and scope

Input: overlapping photographs or a local video of a stationary textured object.
Output: camera poses, sparse colored structure, a dense point cloud and an
optional triangle mesh. Users can upload photos, compare both methods, inspect
progress and per-view diagnostics, interact with geometry and download artifacts.

The primary machine is an Apple Silicon Mac with 16 GB shared memory. Camera
estimation and SGBM run on CPU. Monocular depth is an optional Depth Anything V2
Small stage, using MPS or CPU; it is not another camera-reconstruction method.
There is no multi-view transformer runtime, GPU notebook or corresponding model
installation in the active implementation.

## Pipeline

```text
Photos ─────────────────────┐
Video → automatic frames ───┤
                           ▼
            Orient, validate, resize, intrinsics
                  foreground / alpha masks
                           │
             ┌─────────────┴────────────────┐
             ▼                              ▼
          OUR SfM                       CPU COLMAP
  SIFT / RootSIFT / ORB                SIFT / guided matches
  mutual ratio matching              geometric verification
  normalized eight-point RANSAC      incremental mapping + BA
  conflict-aware feature tracks     largest model + diagnostics
  initialization, triangulation              │
  PnP, local / global BA                     │
             └─────────────┬────────────────┘
                           ▼
       Common OpenCV world-to-camera poses + PLY + observations
                           │
           ┌───────────────┴───────────────────┐
           ▼                                   ▼
     SGBM stereo                     Optional monocular depth
     calibrated pairs                sparse inverse-depth alignment
           └───────────────┬───────────────────┘
                           ▼
         foreground / ROI / multi-view consistency filtering
                           ▼
               TSDF → surface extraction → cleanup
                           ▼
          cloud + mesh viewers / PLY / OBJ / GLB / metrics
```

Both camera methods reconstruct independently. Do not initialize Our SfM from
COLMAP poses, tracks or points. An explicitly disclosed lens-intrinsic prior may
be supplied to both; it is not ground truth. Ground-truth cameras and surfaces
are evaluation references only. Earlier pose-based image selection must be
labelled as such, and must not be called unseen generalization.

## Implementation contracts

- Internal coordinates: OpenCV, world-to-camera, `x_cam = R X + t`, camera center
  `C = -R.T @ t`. Every resize, crop and rotation must update pixel calibration.
- Common artifacts: `poses.json`, `points.ply`, `observations.json`, `summary.json`.
  Preserve image names and record registered and missing views separately.
- Our mathematical core: normalized eight-point estimation, RANSAC, essential
  decomposition, cheirality, triangulation, track handling and sparse robust BA.
  OpenCV supplies feature extraction, descriptor search, homography and PnP;
  SciPy supplies the nonlinear optimizer. Disclose this boundary.
- COLMAP uses pycolmap on CPU in an isolated process. Record camera models,
  calibration locks/refinement, model count, residual statistic and runtime.
- Object profile: conservative foreground handling and bounded matching / BA.
  Maximum quality: full camera frames, up to 12,000 features, exhaustive pairs,
  ranked/uniform eight-point verification for Our SfM, affine/domain-size pooled
  SIFT for COLMAP, and more thorough BA. This costs additional CPU time.
- Capture about 60 sharp, overlapping views covering front, rear, sides and top.
  Uploads support 3–200 photos, at most 25 MB each and 1 GB total. More views do
  not guarantee more accurate cameras or more complete surfaces.
- SGBM requires calibrated, overlapping pairs with usable texture and baseline.
  Monocular depth is an alternative for difficult stereo coverage; relative
  inverse depth needs robust alignment to each method's own sparse landmarks.
- Mesh presets: Balanced 640px / grid 192, High 960px / grid 256, Ultra 1280px /
  grid 384. Never enlarge smaller photos solely to imply greater detail.
- Keep missing geometry visible. A denser grid, triangle cap or display point
  cap must not fabricate observations or imply measured accuracy.

## Sequential experiments

| Stage | Study | Required evidence |
| --- | --- | --- |
| E1 | Our SfM vs. COLMAP on identical inputs | Registration, camera ATE/AUC, observed reprojection, tracks, clouds, time, RSS |
| E2 | No BA / local BA / local + global BA | Fixed photos/config except BA; camera and reprojection changes |
| E3 | Number of views | Same subsets for both methods; registration and pose-error curves |
| E4 | SIFT / RootSIFT / ORB | Other settings fixed; successful and failed conditions |
| E5 | Approximate / calibrated intrinsics / shared-focal BA | Explicit calibration sources and failure to recover K when applicable |
| E6 | Ratio and RANSAC sensitivity | One-factor settings, coverage alongside residuals |
| E7 | Glossy / sparse / blurred / low-texture cases | Measured overlap, track support, failure stages and visual examples |
| M4 | Our mesh against original fountain-P11 GT scan | Camera-only Sim(3), GT-box crop, accuracy, completeness, F-scores, coverage |
| M5 | Our SfM vs. COLMAP meshes | Same depth backend and fusion policy, same aligned rendering viewpoint |

Existing development experiments may be projected into a two-method collection
without rerunning reconstruction. Preserve retained scalar measurements exactly,
record source hashes, regenerate method-comparison figures, and identify the
projection as derived evidence. Do not silently present an old mixed-method plot
as a current two-method result. Raw historical measurements are archival.

Temple has supplied GT cameras. Existing Car/Fountain COLMAP camera references
measure agreement, not independent truth. No Temple GT triangle model was supplied.
Fountain scan completeness includes unseen regions unless visibility masking is
implemented; the current scorer is not the official EPFL visibility-aware scorer.
Camera, landmark-based depth alignment and ROI effects are coupled in M5, so the
comparison is not a strictly pose-only causal experiment. Fusion-held-out photos
still participated in camera estimation; report coverage alongside masked image
scores. Missing ATE stays unavailable and missing camera pairs count against AUC.

## Interface and storage

Four tabs: Workspace, Pipeline & methods, Performance, Experiments.

Workspace offers Our SfM, COLMAP or Compare both. Compare both creates exactly
two serially queued runs from byte-identical uploaded photos. Keep cancel/delete,
progress percentages, dark mode, input thumbnails, point-size control, XYZ hover,
mesh appearance modes and downloads. Existing saved Car/Chair outputs retain
explicit COLMAP + SGBM provenance.

Performance and Experiments show only the active two methods. Include measured
per-run calibration, track support, missing camera reasons, mesh topology, stage
timings and raw-record downloads. No GT scores may be invented for uploads.

Photos and outputs stay under the selected local data directory. Recoverable
delete and immutable prior-attempt snapshots remain. Mesh-only retry may reuse
saved cameras and change depth/mesh quality; input, matching or calibration changes
require a full camera solve. Label reused camera timings separately from the last
attempt. Do not regenerate user-deleted runs without a later explicit request.

## Validation and remaining deliverables

Run the regression suite and verify the native COLMAP subprocess, both-method
creation, unsupported-method rejection, historical-result filtering, image limits,
calibration transforms, robust matching, mesh conventions, consistency and exports.
Test the actual UI: two method cards, both-method upload, graph data/downloads,
cloud/mesh switching, hover coordinates, point size, progress and deletion.

Keep compact test and browser evidence, source/configuration hashes and honest
failure records. Package only current source, small inputs and intended evidence;
exclude raw datasets, weights, virtual environments and obsolete runtimes. Validate
the extracted archive in a fresh directory. Finish the IEEE report, actual author
contribution table, narrated demo and human submission separately.

Useful commands are maintained in README.md and docs/EXPERIMENTS.md. Record actual
execution results in NOTES.md; this plan does not substitute for measured evidence.
