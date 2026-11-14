# Mesh report addendum

This is the mesh-stage report material for integration into the full course report. The final full IEEE report and narrated course demo are separate deliverables.

## Method

Representations: sparse colored points, fused dense colored points, and a vertex-colored triangle mesh. The adapter verifies OpenCV world-to-camera poses. Relative inverse depth from Depth Anything V2 Small is aligned by robustly fitting `1/z_sfm = s*d_rel + b`; SGBM uses the recovered stereo baseline directly. Invalid/edge depths, a sparse ROI and neighboring depth checks filter the maps. TSDF uses voxel length `extent/voxel_div` and truncation four voxels. The running average is `F_new = (w_old F_old + w_obs F_obs)/(w_old+w_obs)`. Marching cubes extracts the zero set; small components and invalid triangles are removed, then a light mesh is decimated for viewing.

## Measured results

| Scene/run | Depth | Full faces | Light faces | Coverage | PSNR dB | SSIM | Watertight |
|---|---|---:|---:|---:|---:|---:|---|
| car_sgbm | sgbm | 119,296 | 80,000 | 0.137 | 17.11 | 0.556 | False |
| chair_sgbm | sgbm | 95,847 | 80,000 | 0.175 | 17.72 | 0.470 | False |
| chair_sgbm_wide_roi | sgbm | 100,870 | 79,999 | 0.181 | 17.85 | 0.474 | False |
| sample_cached | mono | 135,982 | 79,999 | 0.731 | 13.70 | 0.451 | False |
| sample_mono | mono | 136,068 | 80,000 | 0.731 | 13.70 | 0.451 | False |
| sample_sgbm | sgbm | 49,358 | 49,358 | 0.415 | 13.67 | 0.305 | False |
| sphere_cached | mono | 23,648 | 23,648 | 0.421 | 23.09 | 0.986 | False |
| temple_mono | mono | 173,812 | 79,999 | 0.240 | 14.35 | 0.288 | False |

Sources: `validation/mesh/mesh_metrics.csv` and per-run summaries/configs. Times are local measured run times with available model caches, not dependency installation or network download benchmarks.

## Limits and failure study

Temple and the tile fixture use our SfM poses. Chair and car use existing COLMAP poses. Their image-derived surfaces have no supplied GT mesh. Held-out images are excluded from depth/fusion, while their cameras and the sparse scene still come from all-view SfM. Low coverage and image scores must be considered together. Monocular depth smooths/thickens surfaces; SGBM fails on glossy or textureless regions; ROI can crop thin parts. The chair wide-ROI run documents a scene-specific mitigation. `validation/mesh/failure_study/` logs deliberately corrupted depth and too few views on a known analytic sphere.

## Literature and credits

- Curless and Levoy (1996), A Volumetric Method for Building Complex Models from Range Images: https://graphics.stanford.edu/papers/volrange/
- Lorensen and Cline (1987), Marching Cubes: A High Resolution 3D Surface Construction Algorithm, DOI:10.1145/37402.37422.
- Kazhdan, Bolitho and Hoppe (2006), Poisson Surface Reconstruction: https://diglib.eg.org/items/245d6ba1-0209-4365-a4a4-16ce46421620
- Yang et al. (2024), Depth Anything V2: https://arxiv.org/abs/2406.09414
- Depth Anything V2 Small checkpoint (Apache-2.0): https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Open3D (TSDF, Poisson, rendering/viewer), trimesh (exports), scikit-image (SSIM); API use, no copied model implementation.
- Middlebury multiview data: https://vision.middlebury.edu/mview/data/. Other local datasets were user supplied; no GT transforms or meshes were inferred from their filenames.

## Integration assets

The generated PDF addendum uses these recorded numbers and figures. `results/mesh/mesh_segment.mp4` is a 30-second captioned mesh segment for the final demo, not a narrated full course video. No portal upload was performed.
