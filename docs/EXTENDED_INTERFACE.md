# Extended reconstruction interface

Implemented on `codex/extended-reconstruction-interface`, based on `a25c33c`.

The existing upload/reconstruction/viewer workflow remains in Workspace. New
research pages load the versioned experiment evidence already produced by the
project; no reconstruction or benchmark is rerun merely by opening a tab.

## Views

| View | Contents | Source |
| --- | --- | --- |
| Workspace | Uploads, progress, cancel/delete, cloud/mesh, point size, XYZ hover, downloads, saved car/chair | Local workbench jobs and existing geometry |
| Run inspector | Residuals, observed track length, stage timings, topology, growth snapshots, settings, versions | That selected run's own summaries |
| Pipeline & methods | Our SfM / COLMAP / VGGT steps, calibration, dense reconstruction, outputs, limitations, provenance | Implemented pipelines and pinned runtime records |
| Performance / comparison | Filter by scene/method; registration, ATE, AUC and time plots; sortable measurements | E1 |
| Performance / ablations | BA, view-count curves, detectors, intrinsics, matching sensitivity and failure cases | E2–E7 |
| Performance / mesh quality | Accuracy, completeness, F-score, coverage, image diagnostics, aligned renders and interactive meshes | M4/M5 |
| Experiments | Full measured tables, original PNG/PDF plots, CSV/JSON, reports and raw metric records | All nine versioned studies |

Graphs use inline SVG with focusable marks and value readouts. They have no
external chart runtime. The existing Three.js viewer still loads its CDN assets.
Dark mode, responsive layout and keyboard tab navigation apply to the new pages.

## Data interpretation

- 67 recorded method conditions include reused runs and failed attempts; this
  count does not mean 67 independent acquisitions or successful reconstructions.
- Camera ATE is not plotted as zero when alignment is undefined. Zero AUC from
  actual failures stays zero. Partial camera registration is labelled explicitly.
- Car/fountain E1 camera errors are COLMAP agreement. Temple/Dino cameras and the
  M4/M5 original fountain mesh provide the explicitly identified GT references.
- VGGT's camera count means predictions, not feature-verified registration.
  Its exported 50k common points are not its full dense-array count.
- E1 runtime includes method loading/export. Mesh timing excludes upstream
  SfM/VGGT inference and GT evaluation. Process RSS differs from GPU memory.
- Mean COLMAP reprojection residuals remain labelled as means; the inspector
  does not silently convert them into medians.
- New uploads have no supplied GT. Their inspector does not invent camera ATE,
  ground-truth mesh scores or benchmarks against unrelated saved objects.
- See `docs/EXPERIMENT_RESULTS.md` for calibration, development-set, GT-visibility
  and fusion-held-out limitations. Existing saved 60-view car/chair models are
  distinct from the fixed E1 benchmark subsets.

## API and files

`GET /api/research` returns methods, glossary, collection metadata, all recorded
rows and artifact URLs. `GET /api/research/files/<relative-path>` serves only
existing files listed in that collection's evidence manifest, with a resolved
path containment check. It cannot expose arbitrary repository or local files.

`GET /api/jobs/<id>/analysis` and `/api/examples/<name>/analysis` return only that
selected reconstruction's details. Research access is read-only. Existing upload
limits, localhost/Host checks and mutation-origin protections remain in force.

Code: `code/web/research.py`, the routes in `code/web/server.py`, and
`code/web/static/lab.js` / `lab.css`. The small selection event in `app.js` updates
the inspector without changing the geometry or reconstruction backend.

Run with the README command. The preview on port 8766 uses a separate job folder;
the previous service on port 8765 is not stopped by this task. Full validation
and browser checks are recorded in `validation/web_lab/verification.json`.
