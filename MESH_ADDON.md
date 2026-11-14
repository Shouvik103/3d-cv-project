# MESH ADD-ON: dense depth, triangle mesh and mesh view (for an existing SfM project)

Project: 3D Computer Vision course project (IIIT Dharwad), own incremental SfM already implemented and producing camera poses and a sparse 3D point cloud. This document adds **only the mesh**: dense depth, a vertex-colored triangle mesh, an interactive mesh viewer, headless mesh renders, mesh evaluation, and the demo/report/packaging pieces that go with it.

Hard deadline for the whole project: upload by **21 Nov 2026** (brief says "before 22nd"). Today is 14 Nov. Mesh work should be done by about 18 Nov so the rest of the week is for the demo, report and packaging.

Hardware: MacBook Air M4 (Apple Silicon, no CUDA). Free Colab only if something truly needs a GPU (it should not).

---

## 0. Rules for the agent

1. **Do not rewrite or break the existing SfM code.** Add new code under `code/mesh/` and `code/viz/`, new configs under `configs/`. Touch existing files only to add an export (for example, writing `tracks.npz`) or to wire a new CLI flag. Say exactly what you changed in `NOTES.md` and in the README's "what we modified" table.
2. First **inspect the repo** (Section 1) before writing anything. Do not assume file formats.
3. Relative paths only (`pathlib`, resolved from the file location). No hardcoded absolute paths. Everything runs from the repo root.
4. **Never fabricate results.** Every number or figure in the report comes from a logged run saved as JSON/CSV/PNG under `results/`. Save the config and seed with each run.
5. Write the synthetic unit tests (Section 9) before running on real data. Do not move on while they fail. If something fails twice after honest debugging, use the fallback in Section 11, log it in `NOTES.md`, and continue.
6. Keep the zip under 200 MB: no model weights, no raw datasets in the final archive. Small meshes and cached depth for `sample_data/` only.
7. Ask the humans for things only they can provide (Section 12) instead of guessing.

---

## 1. Step zero: inspect and adapt to the existing code

Before writing the mesh code, find out how the existing pipeline stores its results.

1. List the repo. Find where SfM writes its outputs (poses, intrinsics, sparse points, features/tracks) and what the formats are (JSON, NPZ, pickle, PLY, COLMAP text, etc.).
2. Write `code/mesh/adapters.py` with `load_scene(path) -> SceneData`, converting whatever exists into this internal contract (do not change the SfM outputs; convert on load):

```python
@dataclass
class SceneData:
    image_paths: list[Path]          # the same resized images SfM used, in the same order
    K: list[np.ndarray]              # per image 3x3 intrinsics, in pixels of THAT image file
    R: list[np.ndarray]              # per image 3x3, WORLD-TO-CAMERA, OpenCV axes (x right, y down, z forward)
    t: list[np.ndarray]              # per image (3,), x_cam = R @ X + t, camera center C = -R.T @ t
    registered: list[bool]           # which images have valid poses
    points: np.ndarray               # (N,3) sparse points in the same world frame
    colors: np.ndarray | None        # (N,3) uint8, optional
    tracks: dict | None              # optional: point_id -> {image_index: (u, v)}
```

3. **Convention sanity check (mandatory).** If the existing code uses another convention (camera-to-world, OpenGL axes, COLMAP quaternions, etc.), convert it. Then verify: project the sparse points into each registered image and report the **median reprojection error**. If tracks exist, use the observed pixels; otherwise, check that the projected points are mostly in front of the camera and inside the image, and that their projections land on plausible object pixels (save one overlay image per scene). A median reprojection error below about 2 px (with tracks) or a clearly correct overlay is required before continuing. A wrong convention is the single most likely cause of garbage meshes.
4. If no per-point visibility (tracks) is stored, do not stop: the alignment step (Section 3.2) works with "all points that project inside the image in front of the camera" and a robust fit. Optionally add a small export of `tracks.npz` to the SfM code if it is trivial.
5. Record the scene extent: diagonal of the bounding box of the sparse points between the 2nd and 98th percentile. It sets voxel size and the region of interest.

---

## 2. Setup

```bash
source .venv/bin/activate    # use the project's existing environment
pip install open3d torch torchvision transformers trimesh scikit-image pyyaml tqdm
```
- Python 3.11 is the safest for Open3D wheels. Keep your existing versions if they already work; record versions in `NOTES.md`.
- Apple MPS: device `"mps" if torch.backends.mps.is_available() else "cpu"`, and export `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- The monocular depth weights download from Hugging Face on first run (internet needed once). Never put weights in the zip. Cache depth results so graders do not need the model (Section 8).

---

## 3. The mesh pipeline

Code layout:
```
code/mesh/
  adapters.py        # load existing SfM outputs -> SceneData (+ convention check)
  depth_mono.py      # monocular relative depth per image, cached
  depth_sgbm.py      # classical stereo depth (fallback / ablation)
  align_depth.py     # per-image scale/shift fit to sparse points
  filters.py         # depth cleanup, region of interest, multi-view consistency
  tsdf_fusion.py     # TSDF integration + marching cubes
  poisson_mesh.py    # Poisson baseline (ablation)
  mesh_clean.py      # crop, components, degenerate removal, decimation
  mesh_export.py     # PLY (vertex colors), OBJ, GLB
  mesh_eval.py       # metrics
  run_mesh.py        # CLI
code/viz/
  view_mesh.py       # interactive viewer
  render_mesh.py     # headless renders (RaycastingScene)
configs/mesh_default.yaml
tests/test_mesh_*.py
```

### 3.1 Per-image depth (`depth_mono.py`)
- Model: Depth Anything V2 Small via Hugging Face `transformers` (verify the exact model id and its license on the model card; at the time of writing the id is `depth-anything/Depth-Anything-V2-Small-hf`). It outputs **relative inverse depth** (arbitrary scale and shift per image).
- Run on every registered image at `input_long_side = 768` (config), resize the prediction back to the image size, cache to `data/<scene>/cache/depth_mono/<image>.npy` (float16 is fine).
- Process one image at a time to keep memory low.

### 3.2 Scale alignment to the SfM scale (`align_depth.py`)
For each image `i`, take the sparse points visible in it (use tracks if available; otherwise all points that project inside the image with positive depth), read `d_rel` at their pixels, and fit

`1 / z_sfm ≈ s * d_rel + b`

with a **robust** fit (RANSAC on 2 points, then least squares on the inliers; or Huber/IRLS). Here `z_sfm = (R_i X + t_i)_3`. Then metric-in-SfM-units depth is `z = 1 / (s * d_rel + b)`, valid only where `s * d_rel + b > 1e-6`.
- Require at least 30 inlier points; otherwise skip the image and log it.
- Quality gate: median of `|z_fit - z_sfm| / z_sfm` over inliers must be below 0.10; images that fail are excluded from fusion and listed in the log.
- Save `alignment.json` with per-image `s, b, n_inliers, median_rel_residual`.

### 3.3 Depth filtering and region of interest (`filters.py`)
- Drop pixels with large local depth gradients (relative gradient > 0.05, config), depth beyond the 98th percentile of aligned depth (`depth_trunc`), and invalid pixels.
- Region of interest: axis-aligned box around the sparse points (5th to 95th percentile per axis) plus 15% margin. Pixels whose back-projected 3D point falls outside the box are removed. This stops the table or background from fusing into the object.
- **Multi-view consistency (recommended):** back-project pixel `(u,v)` of image `i`, project into 2 neighboring registered images `j` (neighbors by smallest viewing-angle difference), compare the reprojected depth `z_j` with the neighbor's aligned depth `z'`. Keep the pixel if `|z_j - z'| / z_j < 0.02` in at least 2 neighbors (1 if only 1 to 2 neighbors see the point). Config flag so it can be ablated.

### 3.4 TSDF fusion and mesh extraction (`tsdf_fusion.py`)
- `o3d.pipelines.integration.ScalableTSDFVolume`, `voxel_length = scene_extent / voxel_div` (default 256), `sdf_trunc = 4 * voxel_length`, color type RGB8.
- For each kept image: `RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, depth_trunc=..., convert_rgb_to_intensity=False)`, intrinsics `PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)` (scale K if the depth resolution differs from the image resolution), extrinsic = **world-to-camera 4x4** `[R|t]`.
- `vol.extract_triangle_mesh()`, `compute_vertex_normals()`. Also export the dense colored point cloud (`vol.extract_point_cloud()`).
- Colors: OpenCV loads BGR. Convert to RGB before building the Open3D image, and keep colors in the range Open3D expects.
- Poisson baseline (`poisson_mesh.py`, only for the ablation): oriented dense points, `create_from_point_cloud_poisson(depth=8)`, remove vertices below the 5% density quantile.

### 3.5 Cleaning and export (`mesh_clean.py`, `mesh_export.py`)
- Crop to the region of interest; remove degenerate and duplicated triangles/vertices and non-manifold edges; keep connected components with at least 2% of the largest component's triangles (`cluster_connected_triangles`); `remove_unreferenced_vertices`.
- Two versions: **full** (evaluation) and **light** (`simplify_quadric_decimation`, about 80k triangles) for the viewer, demo and `sample_data/`.
- Export `mesh_full.ply`, `mesh_light.ply` (vertex colors), `mesh_light.obj`, `mesh_light.glb` (via `trimesh`). Log vertices, faces, components, boundary edges, surface area, watertightness.
- Vertex colors are the default appearance. A UV texture map is a stretch goal only.

### 3.6 Alternative back-ends (for ablation and as fallback)
- **`sgbm` (classical, Unit I stereo):** for each image pick 2 neighbors with baseline about 5-15% of the scene depth; `cv2.stereoRectify` with our R, t, K; `cv2.StereoSGBM_create(numDisparities=128, blockSize=5, P1=8*3*25, P2=32*3*25, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1, mode=STEREO_SGBM_MODE_SGBM_3WAY)`; reproject with `Q`; undo the rectification rotation; convert to the reference image's depth map and feed the same TSDF fusion. Needs no scale alignment because the baseline `T` already has SfM scale.
- **`vggt` (only if VGGT outputs already exist):** use VGGT's own per-image depth and its own poses, filtered by confidence, through the same fusion code.
- **COLMAP poses (only if a COLMAP result exists):** run the same `mono` back-end using COLMAP's poses and sparse points (convert them into `SceneData` through an adapter).

---

## 4. Mesh view (required)

### 4.1 Interactive viewer (`code/viz/view_mesh.py`)
```
python code/viz/view_mesh.py --mesh <path.ply|obj|glb> [--mode colored|shaded|wireframe|normals]
                             [--poses poses.json] [--points points.ply] [--screenshot out.png]
```
- Load with `o3d.io.read_triangle_mesh`; fail with a clear message if there are no triangles.
- Modes: `colored` (vertex colors), `shaded` (uniform gray + normals), `wireframe` (`o3d.geometry.LineSet.create_from_triangle_mesh`), `normals` (color by normal).
- Optionally overlay camera frustums and the sparse point cloud.
- Show with `o3d.visualization.draw_geometries([...], mesh_show_back_face=True)`; print the mouse/keyboard hints. `--screenshot` writes one image and exits (use the headless renderer if no window is available).

### 4.2 Headless renderer (`code/viz/render_mesh.py`) (reliable on macOS and in scripts)
Use `o3d.t.geometry.RaycastingScene`: `scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))`; `rays = scene.create_rays_pinhole(intrinsic_matrix=K, extrinsic_matrix=ext, width_px=W, height_px=H)` (extrinsic = world-to-camera); `ans = scene.cast_rays(rays)` returns `t_hit`, `primitive_ids`, `primitive_uvs` (barycentric). From those compute:
- **colored** render: interpolate vertex colors with weights `(1-u-v, u, v)`,
- **shaded** render: Lambert with a headlight, `I = max(0, n · (-d))`,
- **normal map** and **depth map** (z-depth = `t_hit * dot(unit_ray, camera_z)`).
Fix the background color and the light direction for every render so methods are comparable. Verify the API names against the installed Open3D version.

Outputs per scene (required): colored orbit, shaded orbit, wireframe close-up, depth or normal render, 2-3 still screenshots for the report. Orbit = camera on a circle around the mesh center at fixed elevation looking at the center, 150-240 frames at 1280x720.

---

## 5. Evaluation (`mesh_eval.py`)

Always align geometry from different sources with a Sim(3) estimated from camera centers (Umeyama) before comparing. If an Umeyama implementation already exists in the repo, reuse it.

| Metric | How | When |
|---|---|---|
| Accuracy / completeness / F-score | sample about 200k points on each mesh (`sample_points_uniformly`), KD-tree distances, `F = 2PR/(P+R)` at τ = 1% of the GT bounding-box diagonal (also 0.5% and 2%). Crop the estimate to the GT bounding volume first | only where a GT mesh exists: Strecha fountain-P11 or Herz-Jesu-P8 provide a ground-truth triangle model and bounding volume on their download page |
| Mesh statistics | vertices, faces, connected components, boundary edges, watertight yes/no, area, mean edge length | all scenes |
| Held-out-view check | fuse without 2-3 held-out registered images, render the mesh from their poses with `RaycastingScene`, compare to the real photo on pixels the mesh hits: PSNR, SSIM (`skimage`), MAE, plus coverage | all scenes |
| Cross-method agreement | Chamfer and F-score between meshes of different methods after alignment (agreement, not truth) | only if COLMAP or VGGT results exist |
| Runtime and memory | depth inference, alignment, fusion, extraction, total | all |

Honesty notes to put in the report: monocular-depth meshes inherit the network's smoothing and thickening; when comparing methods, ours vs. COLMAP share the same depth network and isolate pose/structure quality, while VGGT uses its own depth, so differences involving VGGT mix pose and depth-source effects; Middlebury does not distribute ground-truth geometry, so it supports qualitative meshes only; on self-captured scenes there is no ground truth.

---

## 6. Experiments

| # | Experiment | Priority |
|---|---|---|
| M1 | Mesh from our SfM on the main scene S1 and at least one more scene: colored, shaded and wireframe renders, mesh statistics, viewer works | **Must** |
| M2 | Held-out-view check (PSNR/SSIM/coverage) on S1 and the second scene | Should |
| M3 | Back-end ablation: `mono` vs. `sgbm`; TSDF vs. Poisson; voxel size (extent/128, /256, /384); consistency filter on/off | Should |
| M4 | Mesh on a ground-truth scene (fountain-P11): accuracy, completeness, F-score, error colormap | Should (needs the GT mesh files) |
| M5 | Mesh from COLMAP poses and from VGGT depth with the same back-end, side-by-side comparison | Nice (only if those results already exist) |
| M6 | Failure study: glossy or textureless region, too few views, thickened surfaces, background fusion, a wrong-scale image | **Must** (qualitative, a few numbers) |

---

## 7. Capture advice for new photos (if humans retake any scene)
Cover the object from all sides and add a second ring from a higher elevation (30-45 degrees looking down) so tops are visible. Soft, even light; matte, non-transparent, non-black object; plain non-reflective surface under it; keep the object mostly in frame; avoid blur and zoom changes. If SfM has already been run on the existing photos, reuse them: the mesh stage only needs them plus the existing poses.

---

## 8. Packaging additions (assignment rules still apply)

The submission zip must keep its required structure (`report.pdf`, `demo.mp4`, `README.md`, `requirements.txt`, `code/`, `sample_data/`) and stay under 200 MB.
- `sample_data/<scene>/`: images (small), the SfM outputs for them, **cached depth** (`depth_mono/*.npz`, small resolution, compressed), and the **light meshes** (`mesh_light.ply` and `.glb`, a few MB each). Keep `sample_data/` under about 40 MB. Graders can then run `run_mesh.py` and `view_mesh.py` without downloading the depth model or having a GPU.
- `requirements.txt`: add `open3d`, `torch`, `torchvision`, `transformers`, `trimesh`, `scikit-image` (only what `code/` really imports; pin versions after it works).
- README additions:
```
## Mesh and mesh view
python code/mesh/run_mesh.py --scene sample_data/scene_small --out outputs/mesh --config configs/mesh_default.yaml
python code/viz/view_mesh.py --mesh outputs/mesh/mesh_light.ply
python code/viz/view_mesh.py --mesh sample_data/scene_small/mesh_light.ply --mode shaded
python code/viz/render_mesh.py --mesh sample_data/scene_small/mesh_light.ply --out outputs/mesh_renders
(the first real depth run downloads the small depth model from Hugging Face; sample_data ships cached depth)
Meshes also open in MeshLab or Blender.
```
- README "Third-party code and modifications" table: add Depth Anything V2 Small (inference only), Open3D (TSDF, Poisson, RaycastingScene, viewer), trimesh (export), scikit-image (SSIM), and describe our own contribution: scale alignment, filtering, fusion pipeline, cleaning, metrics, viewer.
- Cite all outside code in the report.
- Fresh-machine test: unzip into a new folder, create a new venv, install, run the mesh viewer on the shipped mesh and `run_mesh.py` on the sample. Check `grep -rn "/Users/" code/` finds nothing.

---

## 9. Unit tests (write first; `pytest tests/test_mesh_*.py`)

Build a synthetic scene: a sphere of radius 1 (and a box) at the origin; 12 cameras on a sphere of radius 3 around it, looking at the center, known K, exact analytic depth maps and colors from ray-sphere intersection.

| Test | Pass criterion |
|---|---|
| Depth alignment | apply a known per-image scale and shift to true inverse depth, add 5% outlier points; recovered `s, b` within 2%; aligned depth within 2% of truth |
| TSDF fusion on the sphere/box with exact depth | mesh vertices within 1.5 voxels of the true surface (median < 0.5 voxel); one connected component |
| TSDF pose convention | feeding camera-to-world instead of world-to-camera must produce a clearly wrong mesh (this test guards the convention) |
| `RaycastingScene` renderer | rendered sphere depth matches analytic depth within 1e-3 relative; normals within 2 degrees in the interior |
| Cleaning | injected small floating blobs are removed; decimated mesh keeps Chamfer distance to the full mesh below 0.5% of the bounding-box diagonal |
| Export round trip (PLY, OBJ, GLB) | same vertex count and colors within 1/255 after reload |
| Viewer smoke test | `view_mesh.py --screenshot` writes a non-empty image |
| Real data smoke test | on `sample_data`, `run_mesh.py` finishes in under about 15 minutes on the M4 (under 3 minutes with cached depth), the light mesh loads in the viewer |

---

## 10. Code skeletons (adapt names to the repo)

Robust inverse-depth alignment:
```python
def fit_inverse_depth_affine(d_rel, z_sfm, iters=200, tol=0.05, seed=0):
    """Fit 1/z_sfm ~ s*d_rel + b robustly. Returns (s, b, inlier_mask, median_rel_residual)."""
    w = 1.0 / z_sfm
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(w), 2, replace=False)
        if abs(d_rel[i] - d_rel[j]) < 1e-6: continue
        s = (w[i] - w[j]) / (d_rel[i] - d_rel[j]); b = w[i] - s * d_rel[i]
        pw = s * d_rel + b
        ok = pw > 1e-6
        z_fit = np.where(ok, 1.0 / np.maximum(pw, 1e-6), np.inf)
        inl = ok & (np.abs(z_fit - z_sfm) / z_sfm < tol)
        if best is None or inl.sum() > best[2].sum(): best = (s, b, inl)
    s, b, inl = best
    A = np.stack([d_rel[inl], np.ones(inl.sum())], 1)
    s, b = np.linalg.lstsq(A, w[inl], rcond=None)[0]
    z_fit = 1.0 / np.maximum(s * d_rel[inl] + b, 1e-6)
    return s, b, inl, float(np.median(np.abs(z_fit - z_sfm[inl]) / z_sfm[inl]))
```

TSDF fusion:
```python
import open3d as o3d, numpy as np
def fuse(frames, voxel_length, sdf_trunc, depth_trunc):
    # frames: dicts with color (uint8 HxWx3 RGB), depth (float32 HxW, SfM units), K, R, t (world-to-camera)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for f in frames:
        H, W = f["depth"].shape
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, f["K"][0,0], f["K"][1,1], f["K"][0,2], f["K"][1,2])
        ext = np.eye(4); ext[:3,:3] = f["R"]; ext[:3,3] = f["t"]
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(f["color"])),
            o3d.geometry.Image(np.ascontiguousarray(f["depth"].astype(np.float32))),
            depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, ext)
    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    return mesh, vol
```

Cleaning:
```python
def clean(mesh, roi_box, min_component_frac=0.02, light_target=80000):
    mesh = mesh.crop(roi_box)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges()
    cl, n_per, _ = mesh.cluster_connected_triangles()
    cl = np.asarray(cl); n_per = np.asarray(n_per)
    keep = n_per >= min_component_frac * n_per.max()
    mesh.remove_triangles_by_mask(~keep[cl]); mesh.remove_unreferenced_vertices()
    light = mesh.simplify_quadric_decimation(target_number_of_triangles=light_target)
    for m in (mesh, light): m.compute_vertex_normals()
    return mesh, light
```

Headless render outline:
```python
def render(mesh, K, R, t, W, H, mode="colored"):
    tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(tm)
    ext = np.eye(4); ext[:3,:3] = R; ext[:3,3] = t
    rays = scene.create_rays_pinhole(intrinsic_matrix=K, extrinsic_matrix=ext, width_px=W, height_px=H)
    ans = scene.cast_rays(rays)
    hit = np.isfinite(ans["t_hit"].numpy())
    tri = ans["primitive_ids"].numpy(); uv = ans["primitive_uvs"].numpy()
    # colored: interpolate vertex colors with (1-u-v, u, v) over the triangle's 3 vertices
    # shaded: I = clip(dot(n, -ray_dir), 0, 1); depth: z = t_hit * dot(ray_dir, cam_z)
    # return HxWx3 uint8 with a fixed background color, and the hit mask
```

Config `configs/mesh_default.yaml`:
```yaml
seed: 0
depth_backend: mono          # mono | sgbm | vggt
mono:
  model_id: depth-anything/Depth-Anything-V2-Small-hf   # verify id and license on the model card
  input_long_side: 768
  device: auto
align:
  min_points: 30
  robust: ransac
  max_median_rel_residual: 0.10
filter:
  depth_trunc_pct: 98
  edge_grad_thresh: 0.05
  consistency: true
  consistency_rel_tol: 0.02
  consistency_min_views: 2
  roi_percentiles: [5, 95]
  roi_margin_frac: 0.15
tsdf:
  voxel_div: 256
  sdf_trunc_voxels: 4
poisson: {depth: 8, density_quantile: 0.05}
clean: {min_component_frac: 0.02, light_target_triangles: 80000}
export: [ply, obj, glb]
heldout: {num_views: 3, seed: 0}
```

---

## 11. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Existing pose convention is different or unclear | do the convention check in Section 1.3; fix in `adapters.py`, not in the SfM code |
| Depth model cannot be downloaded or is too slow | lower `input_long_side`; run on Colab and copy the cached depth back; or use the `sgbm` back-end; state which was used |
| Doubled surfaces or ghosting | alignment failed for some images: check `alignment.json`, drop bad images, tighten the consistency filter, re-verify the world-to-camera extrinsics |
| Table or background fused into the object | tighten the region-of-interest box, lower `depth_trunc_pct`, crop the mesh |
| Many holes | fewer views or angles: add a higher ring if retaking photos; enlarge `sdf_trunc`; coarser voxels; report as a limitation |
| Mesh too smooth or blobby | raise `input_long_side`, use finer voxels, keep the consistency filter; discuss as a monocular-depth limitation |
| Open3D window does not open | use `render_mesh.py`; export GLB/OBJ and open in MeshLab or Blender; document in the README |
| `RaycastingScene` API mismatch | check the installed Open3D version's tensor API names; fall back to `draw_geometries` screenshots or a simple z-buffer rasterizer |
| Out of memory in TSDF | larger voxels (extent/192), fewer or smaller frames, tighter ROI |
| Colors wrong | BGR vs. RGB; colors scaled 0-255 vs. 0-1 |
| Mesh flipped or inside-out | wrong pose convention or normal orientation; run the synthetic sphere test first |
| Time overrun | keep M1 and M6 (mesh views and failure study); drop M5, then M4/M3 |

---

## 12. Schedule (mesh work only)

| Day | Date | Goal | Gate |
|---|---|---|---|
| 1 | Sat 14 / Sun 15 | Inspect repo, `adapters.py`, convention check, synthetic mesh tests, depth model running, alignment working | first colored mesh from S1 appears in `view_mesh.py` |
| 2 | Mon 16 | Filters, cleaning, light mesh, headless renderer, export, tests green | shaded, colored and wireframe renders for S1 |
| 3 | Tue 17 | Second scene, held-out check, ablations (M3), GT-mesh evaluation if files are available (M4) | result JSONs and figures saved |
| 4 | Wed 18 | Orbit video segments, report figures and text (Section 13), `sample_data` meshes and cached depth, README and requirements updates | mesh segment ready for the demo |
| 5-7 | Thu 19 - Sat 21 | Integration into `demo.mp4`, final report, fresh-machine test, upload | submitted (do not rely on the 21st) |

Humans must provide: the path to the existing SfM outputs and images and which scenes to mesh; confirmation of the convention if unclear; the GT mesh and bounding-volume files if fountain-P11 is used (download from the EPFL page: https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html); any COLMAP or VGGT outputs that already exist; and the upload and portal tagging at the end.

---

## 13. Report and demo additions

**Methodology (Section V):** add a paragraph and one small figure on the mesh stage: monocular depth (relative inverse depth), robust per-image scale/shift alignment to the sparse structure, filtering, TSDF fusion with voxel size `scene_extent/256` and truncation `4 * voxel`, marching cubes, cleaning and decimation; give the equation `1/z_sfm ≈ s d_rel + b` and the TSDF running average. State the 3D representations used: sparse point cloud, dense point cloud, triangle mesh with vertex colors.

**Results (Section VI):** Fig. A mesh views for S1 (colored, shaded, wireframe close-up); Fig. B comparison with the GT-mesh error colormap or, if no GT, two scenes side by side; Table mesh stats and held-out PSNR/SSIM/coverage, F-score where GT exists; mesh ablation table (M3); failure panel (M6). Interpret limitations honestly.

**Objectives (Section IV):** add the bullet: build and evaluate a vertex-colored triangle mesh from the recovered poses with a TSDF back-end, and provide an interactive mesh viewer and rendered views.

**Literature review (Section III):** add TSDF fusion (Curless and Levoy 1996), marching cubes (Lorensen and Cline 1987), Poisson surface reconstruction (Kazhdan et al.), monocular depth foundation models (Depth Anything V2) with their limitations (scale ambiguity, smoothing). Verify each bibliographic entry by search before citing.

**References and credit:** cite Open3D, Depth Anything V2, trimesh, scikit-image and any dataset used.

**Demo (about 30 s mesh segment):** colored mesh orbit, then shaded, then wireframe close-up, then side-by-side meshes if COLMAP/VGGT results exist, with the viewer screen recording if available. Render frames with `render_mesh.py` and assemble with `ffmpeg -framerate 30 -i frames/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 23 mesh_segment.mp4`. Keep the final `demo.mp4` between 1 and 2 minutes and under about 25 MB.

---

## 14. Definition of done

- Synthetic mesh tests pass.
- A vertex-colored mesh of S1 and at least one more scene is produced from our poses; `view_mesh.py` opens it; shaded, colored and wireframe renders exist.
- Held-out-view metrics and mesh statistics are logged as JSON; GT-mesh metrics if the files were available.
- `sample_data/` contains light meshes and cached depth; README, requirements, and the third-party credits table are updated.
- The report has the mesh methodology, figures, table and limitations; the demo includes a mesh segment.
- Fresh-machine test passes: unzip, install, view the shipped mesh, run `run_mesh.py` on the sample, no hardcoded paths, total under 200 MB.
