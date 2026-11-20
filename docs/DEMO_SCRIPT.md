# Reconstruction Lab demo script

Record the local interface at http://127.0.0.1:8765/#workspace. Aim for
1 minute 55 seconds to 2 minutes and export as `demo.mp4` at 1080p.

## Preparation

Load Chair and Hotdog once so the models open quickly. Start with Chair's point
cloud selected. Locate the Temple comparison and the E5/E7 experiment views.
Rehearse the narration and allow time for navigation.

## Recording flow

| Time | Screen action | Narration |
| --- | --- | --- |
| 0:00–0:15 | Slowly orbit Chair's point cloud in Workspace. | This is Reconstruction Lab, our image-based 3D reconstruction project. It takes overlapping photographs of a stationary object, estimates camera poses, and produces a colored point cloud and an optional triangle mesh. |
| 0:15–0:30 | Show Reference, close it, switch to Triangle mesh and orbit. | Here is an original input photograph, followed by the reconstructed surface. We can rotate, zoom and inspect the geometry, switch its appearance, and download the result as PLY, OBJ or GLB. |
| 0:30–0:48 | Show the method, depth and quality controls. | The workspace offers our incremental Structure from Motion implementation, CPU COLMAP, or a comparison using identical photos. Both camera methods support SGBM stereo or optional monocular depth, followed by filtering and TSDF surface fusion. |
| 0:48–1:05 | Open Pipeline & methods and show both cards. | Our implementation includes eight-point estimation, RANSAC, pose recovery, triangulation, feature tracks and bundle adjustment. OpenCV provides feature extraction and PnP, while SciPy provides nonlinear optimization. COLMAP runs independently as the baseline. |
| 1:05–1:25 | Open Performance → Method comparison and select Temple. | We evaluated the methods through nine studies. On calibrated Temple, our method reconstructed all sixteen cameras, while COLMAP reconstructed thirteen. COLMAP achieved lower position error on its reconstructed cameras, so we interpret error alongside registration coverage and pose AUC. |
| 1:25–1:43 | Open Ablations & failures. Show E5 calibration, then E7. | Our experiments examine bundle adjustment, photo count, feature detectors, calibration and matching thresholds. Temple’s position error increased substantially with approximate calibration. Sparse views, blur and difficult surfaces also caused partial reconstructions or failures, which remain visible. |
| 1:43–2:00 | Return to Workspace, select Hotdog and orbit its mesh. | The interface also reports reconstruction growth, stage timings and mesh topology. These results demonstrate an end-to-end reconstruction workflow with measurable limitations. Missing views and uncertain geometry can leave holes, and our evaluation distinguishes ground-truth accuracy from agreement with COLMAP. |

## Export checks

The recording shows saved results. Keep registration coverage visible when
discussing camera error. Play the exported MP4 from beginning to end and check
clear audio, readable text, successful model loading and a duration of 1–2 minutes.

Temple figures come from the retained E1 measurements in
`validation/experiments_20261119_classical/E1/results.json`. Temple has supplied
GT cameras. Car and fountain camera measurements use COLMAP references.
