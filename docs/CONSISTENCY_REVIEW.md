# Project consistency review

Reviewed 20 November 2026 against `docs/ACTIVE_PIPELINE.md`, the assignment brief
in `AGENT_PIPELINE.md`, and `MESH_ADDON.md`. The active scope is Our SfM and CPU
COLMAP, with optional monocular depth. This review checks the current working
tree, including the original-photo Reference button.

## Implementation and interface

| Requirement | Current evidence and outcome |
| --- | --- |
| Two independent camera methods | Live capabilities expose Our SfM and COLMAP. Compare both creates two serially queued child runs from identical uploads. API regression tests cover this behavior. |
| Mathematical implementation | Source implements normalized eight-point estimation, RANSAC, essential decomposition, triangulation, tracks and BA residuals. OpenCV supplies features, matching primitives, homography checks and PnP. SciPy supplies optimization. The interface now explicitly describes this boundary. |
| Calibration and coordinates | World-to-camera geometry, resized/cropped intrinsics, depth alignment and fusion conventions have regression coverage. |
| Dense geometry | SGBM and optional Depth Anything V2 Small feed filtering and TSDF. PLY, OBJ and GLB exports and mesh topology have regression coverage. |
| Runtime and quality controls | The 200-photo cap, 25 MB image cap, 1 GB run cap and Balanced/High/Ultra settings agree between the plan, implementation and interface. |
| Workspace | Uploads, progress, cancellation, recoverable deletion, retries, original-photo reference, cloud/mesh viewing, appearance controls and downloads are wired to local jobs. |
| Saved library | Live API returns Chair and Hotdog, each reconstructed by Our SfM and COLMAP. All four are completed. These individual runs have no linked comparison group. |
| Your comparisons | Shows linked Compare both groups. The empty selector now states that no comparisons are available, is disabled, and clears stale results. |
| Video | Automatic frame selection and sparse Our SfM reconstruction are CLI features. The interface correctly identifies photo uploads and the separate video command. Real handheld-video validation remains outstanding. |
| Public archive | A fresh build contains four selected runs and nine studies. Original uploaded photos remain local, so the Reference control stays hidden in this edition. |

## Experiment and metric consistency

The active collection is `validation/experiments_20261119_classical`. All 242
manifest files match their recorded sizes and SHA-256 hashes. The nine studies
contain 52 rows, including failures and reused conditions.

| Studies | Coverage |
| --- | --- |
| E1 | Two-method comparison on car, fountain and Temple |
| E2–E6 | BA, photo count, features, calibration and matching sensitivity |
| E7 | Difficult surfaces, sparse input, blur and low texture |
| M4–M5 | Ground-truth mesh evaluation and camera-source comparison |

The collection derives from retained historical measurements. This audit does
not rerun reconstruction experiments or establish new accuracy scores. Temple
camera errors use supplied ground truth. Car/fountain camera errors use COLMAP
agreement. M4/M5 use the original fountain scan, with the documented crop and
visibility limitations. Uploads do not receive invented GT scores. The glossary
now distinguishes benchmark median residuals from per-upload mean/median values.

The experiments expose limitations of the implementation. Partial registration,
initialization failures, calibration failures and incomplete surfaces remain in
the measured results. Existing settings were tuned on development inputs, so
these results do not establish performance on unseen scenes.

## Checks performed

- Complete regression suite passed with 85 tests in 49.24 seconds. One existing
  Dash pytest-plugin deprecation warning remains.
- Live capabilities, job library and research APIs agree with the recorded scope.
- JavaScript syntax checks passed for the application and research interface.
- Focused JavaScript checks passed for empty comparisons, repopulation and
  clearing obsolete results.
- A fresh static archive build passed, with all referenced model and research
  files present.
- Git whitespace validation passed.

The Reference preview was browser-verified during its implementation. This audit
used source review, APIs, build checks and regression tests. A fresh visual pass
through every control was not performed.

## Remaining course deliverables

The reconstruction implementation and experiment interface align with the active
scope. Overall course submission is incomplete. `TEAM_CONTRIBUTIONS.md` still
contains placeholders. The mesh report/addendum and short mesh segment are
supporting materials. The final 4–6 page IEEE report, actual author details and
contributions, full 1–2 minute demo, final archive under 200 MB, extracted-archive
reproduction check and portal submission still require completion or verification.

No new benchmark runs or public deployments were performed in this review.
