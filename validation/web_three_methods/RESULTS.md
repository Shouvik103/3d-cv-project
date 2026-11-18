# Fresh same-upload reconstructions

Nine new jobs were created, uploaded and started through the same HTTP API as the site. Each method has its own reconstruction, geometry, log, measurements and downloadable outputs.

| Object | Method | Cameras | Mesh triangles | Status | Model |
| --- | --- | ---: | ---: | --- | --- |
| Temple | ours | 16/16 | 79,999 | completed | [Interactive mesh](temple/ours/interactive_mesh.html) |
| Temple | colmap | 13/16 | 80,000 | partial | [Interactive mesh](temple/colmap/interactive_mesh.html) |
| Temple | vggt | 16/16 | 80,000 | completed | [Interactive mesh](temple/vggt/interactive_mesh.html) |
| Car | ours | 4/30 | 80,000 | partial | [Interactive mesh](car/ours/interactive_mesh.html) |
| Car | colmap | 10/30 | 79,999 | partial | [Interactive mesh](car/colmap/interactive_mesh.html) |
| Car | vggt | 30/30 | 8,575 | completed | [Interactive mesh](car/vggt/interactive_mesh.html) |
| Chair | ours | 20/30 | 66,775 | partial | [Interactive mesh](chair/ours/interactive_mesh.html) |
| Chair | colmap | 30/30 | 60,095 | completed | [Interactive mesh](chair/colmap/interactive_mesh.html) |
| Chair | vggt | 30/30 | 79,999 | completed | [Interactive mesh](chair/vggt/interactive_mesh.html) |

The nine jobs are listed in Workspace → Your reconstructions at http://127.0.0.1:8766. Performance → Your comparisons shows the same-image groups, previews, charts, sortable measurements and CSV/JSON/PNG exports.

Temple uses all 16 source views; car and chair use 30 covering views each. SHA-256 checks verify identical prepared photo bytes across methods within each group. Temple uses supplied input K. Car/chair use earlier COLMAP intrinsics estimates and pre-existing image selections, but no earlier poses, tracks, point clouds, depths or meshes were injected. Original chair alpha is retained for filtering.

All nine have point clouds and triangle meshes. Partial registration remains explicit: camera coverage is not completed by inventing poses. VGGT camera counts mean predictions rather than feature-verified registrations. Models are partial and non-watertight. Different depth backends and camera sets affect the surface and point counts.

ATE/AUC in these comparisons are agreement with that upload’s fresh COLMAP model, not GT. Reprojection means/medians remain labelled. Job time includes preparation/inference/mesh/export; caches and settings vary, so it is not a controlled speed benchmark.

Initial failed/partial stereo attempts, the native-threading stall, and later matching-threshold retries are retained in per-job attempt archives. Retry controls now start with the selected run’s saved settings, independently of the upload form. The current test suite passes 68 tests.

Raw local output root: `outputs/web_ui_extended`. Metadata and compact rendered/interactive evidence are retained here; uploaded photos, large raw arrays and full geometry are not copied into Git.
