# M4 — fountain-P11 mesh study

| Source | Status | Fused views | Faces | Accuracy % | Completeness % | F@0.5% | F@1% | F@2% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ours | complete | 8 | 8632 | 3.205 | 10.087 | 0.028 | 0.059 | 0.140 |

## Protocol and limits

- Real original EPFL fountain GT triangle mesh and supplied camera files; hashes and downloads are recorded. Sim(3) fits shared camera centers only, followed by GT-bounding-box crop; no mesh ICP.
- 200,000 area-uniform samples in each direction, seed 0. Accuracy is estimate→GT distance; completeness is GT→estimate distance (lower is better). F-score thresholds are fractions of GT diagonal. This is not the official EPFL visibility-aware evaluation.
- GT is the original 26M-face mesh, sampled without geometry simplification. The displayed GT and vertex-error colors use those 200k reference samples, so distances include sampling approximation.
- Completeness covers the full GT mesh, including scan regions that may not be visible in all input images or retained by the reconstruction ROI. No visibility mask is applied; this makes it a strict full-scan comparison.
- Ours and COLMAP use the same pinned Depth Anything V2 Small model, 384px inference, image-only depth cache, affine depth alignment gates, filtering and TSDF parameters (voxel divisor 192). Each uses its own E1 estimated cameras and sparse landmarks. Consequently camera, landmark-alignment and ROI effects are coupled; this is not a pose-only causal experiment.
- VGGT uses its own jointly predicted cameras, intrinsics and depth, official 518px preprocessing, and the same filtering/TSDF policy; its confidence gate is recorded. Different depth models/resolutions are a confound, not an isolated pose comparison.
- Same three image names are withheld from fusion when all eleven views register. They participated in SfM/VGGT camera estimation; VGGT also jointly inferred their RGB/depth. These are fusion-held-out rendering diagnostics, not strictly unseen-image reconstruction tests.
- PSNR/SSIM are measured only where geometry renders; coverage is reported so incomplete meshes cannot appear successful from a small good-looking patch. Meshes may be partial and non-watertight.
- GT geometry is used only for evaluation, never for depth alignment, meshing, or selecting reconstruction settings. No parameters were tuned against its scores.

Source: [EPFL original fountain model and research-use notice](https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/modelFountain.html). Cite Strecha et al., *On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery*, CVPR 2008.
