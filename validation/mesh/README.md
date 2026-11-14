# Measured mesh evidence

JSON/CSV rows and figures are copied from actual local `results/mesh/` runs by
`code/mesh/collect_results.py`. Large dense clouds, raw images and model weights
are excluded. `sample_ours` / `sample_*` use our own SfM on the original generated
tile fixture; `temple_mono` uses our own 16-view Temple reconstruction. Chair and
car use their existing COLMAP camera estimates, explicitly not our SfM poses.

Held-out metrics exclude the selected views from depth generation and fusion,
but poses/sparse structure were estimated with all views. PSNR and SSIM are over
render hits (SSIM requires fully hit windows); coverage must be read alongside
these scores. No real GT triangle mesh or VGGT depth was supplied. No real GT
mesh F-score or VGGT mesh comparison is claimed. Geometry comparison CLI supports
them if supplied later.

`ablations_sample/` contains measured own-SfM tile-scene backend, voxel,
consistency and Poisson runs. `ablations_fixture/` uses analytic cached inverse
depth, explicitly not network predictions. `failure_study/` deliberately corrupts
analytic sphere depths to show fusion failure. Real chair/car meshes also retain
holes and surface artifacts; a non-watertight result is not hidden or filled.

`fresh_machine.json` records an isolated new venv, relocated sources, offline
sample reconstruction, headless viewer and mesh tests. `archive_check.json`
records the actual archive extraction check when present. `tests.txt` is the
whole-project pytest log. Config/seed/library versions accompany every run.
