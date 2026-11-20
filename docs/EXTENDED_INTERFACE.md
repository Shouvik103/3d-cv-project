# Extended reconstruction interface

The interface is integrated into `main`. Active reconstruction methods:
**Our SfM** and **CPU COLMAP**. Workspace offers either method or **Compare both**,
which creates exactly two independent, serially queued jobs from identical uploads.

## Views

| View | Contents | Source |
| --- | --- | --- |
| Workspace | Uploads, progress %, cancel/delete, cloud/mesh, point size, XYZ hover, downloads, saved runs and original-photo reference | Local jobs and existing geometry |
| Run inspector | Residuals, tracks, calibration provenance, per-view registration diagnostics, topology and stage times | The selected run's own summaries |
| Pipeline & methods | Two camera pipelines, SGBM / monocular depth, TSDF, limitations | Current implementation |
| Performance / Your comparisons | Same-upload jobs, previews, 3D links, tables, graphs and CSV/JSON/PNG downloads | Compare both uploads |
| Performance / Camera comparison | Registration, ATE/AUC, residuals and time | E1 classical measurements |
| Performance / Ablations | BA, views, detector, calibration, matching and failures | E2–E7 |
| Performance / Mesh quality | Accuracy, completeness, F-scores, coverage and common-view renders | M4/M5 |
| Experiments | Tables, plots, reports and metric downloads | Nine retained classical studies |

Graphs use focusable SVG marks and value readouts. The Three.js geometry viewer
supports photo colors, shading, wireframe and normals. Dark mode, keyboard tabs
and responsive layout apply throughout.

## Reconstruction settings

All choices accept 3–200 photos, 25 MB per photo and 1 GB total. About 60 sharp,
overlapping views covering every side are recommended. Object profile bounds
matching and BA; Maximum quality uses full camera frames, exhaustive matching,
more features and more thorough refinement. Matching resolution is independent
of depth / TSDF quality. Smaller originals are not enlarged.

SGBM stereo is the upload default. Optional monocular depth uses the pinned
Depth Anything V2 Small model, aligned to each method's own sparse landmarks.
Both feed the same filtering / TSDF stage; they are depth choices, not additional
camera-reconstruction methods. Mesh-only retry can reuse cameras; changes to
calibration, matching or input preparation require a full solve. Prior attempts
remain locally, and timing fields identify reused camera solves.

## Interpretation and evidence

The active collection is `validation/experiments_20261119_classical`, derived
from retained 18 November results. Metrics for the retained methods are unchanged;
comparison reports and plots are regenerated. Source hashes and scope are in
`derivation.json`. This is a projection of existing evidence, not a new benchmark.
Historical mixed-method records remain archival and are not exposed in the UI.

Camera ATE/AUC against COLMAP measure agreement, not GT accuracy. Temple has GT
cameras; M4/M5 use the original fountain scan. Undefined alignment stays missing;
failures and partial registration remain visible. Low reprojection error alone
cannot establish correct global geometry. Mesh scoring is not the official EPFL
visibility-aware scorer. Fusion-held-out photos participated in camera estimation.

Uploads have no supplied GT cameras or surfaces, so their inspector does not
invent GT scores. The current local library contains Chair and Hotdog runs from Our SfM and COLMAP,
each with 200 photos and SGBM depth. The removed 60-view Car/Chair example cards
are historical outputs. The public archive retains four selected snapshot runs.
Your comparisons lists linked Compare both uploads. With no linked groups, its
selector is disabled and explains how to create one. Independently saved runs
remain in Workspace. The local Reference button previews an original uploaded photo chosen for the run.

## API

- `GET /api/capabilities`: two supported methods, comparison method list, 200-view cap.
- `GET /api/jobs`: supported local jobs only; existing retired-method outputs stay
  on disk but cannot be started, retried or opened through active job routes.
- `GET /api/research`: active methods, glossary, projected measurements and assets.
- `GET /api/research/files/<path>`: manifest-listed files in the active collection,
  guarded by resolved containment checks.
- `GET /api/jobs/<id>/analysis`: this reconstruction's measurements and diagnostics.
- `GET /api/comparisons/<id>` and `/files/comparison.csv|json|png`: two-method reports.
- `POST /api/jobs/<id>/retry`: explicit settings with preserved prior attempts.

Localhost/Host checks, mutation-origin protections and queue generation tokens
remain. User-deleted runs are not regenerated. Launch instructions are in README.md.
