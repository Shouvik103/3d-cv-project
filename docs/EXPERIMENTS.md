# Sequential experiment record

Execution order: E1 → E2 → E3 → E4 → E5 → E6 → E7 → M4 → M5.
An experiment is not marked complete merely because its code exists. The live
ledger is `results/experiments_20261118/experiment_status.json`; raw method logs,
configuration, performance, input hashes and evaluations accompany each result.

## E1 protocol, frozen before method runs

`configs/experiments.yaml` defines car S1 (20 uniformly spaced views from the
existing curated 60), fountain S2 (all 11), and calibrated Middlebury Temple
(all 16). Every method receives the same prepared image bytes and filenames.
Our SfM and COLMAP use identical fixed per-scene PINHOLE calibration.
Classical images have maximum side 1200; smaller inputs are not enlarged.
Image sets and calibration are hashed and cannot silently change on resume.

Temple provides GT cameras. Car and fountain comparisons use existing full-data
COLMAP poses with refined intrinsics as pseudo-references, not ground truth.
Existing car curation used COLMAP camera estimates; the relaxed SfM config was
previously tuned on Temple. Consequently these are development results, not an
unseen benchmark or evidence of universal performance. Partial registration and
failures remain in the output; pose AUC penalizes missing registrations.

Report pose errors/AUC, registered/total views, reprojection errors where observed
tracks exist, point count, track length, elapsed time and peak process RSS.
Point-cloud Chamfer/F-score are explicitly sparse-COLMAP agreement, not true
geometry accuracy. Those cloud scores use identical sample counts and camera
Sim(3) alignment. There is no supplied Temple GT triangle model.

Run the two CPU methods with `.venv/bin/python code/experiments/run_e1.py`.
Workers record end-to-end time including image processing and export; caches and
hardware must be considered when interpreting timing.

## E2 — BA ablation

Run `code/experiments/run_ablation.py --experiment E2`. It changes only BA mode
(none, local, local+global) on the fixed car and Temple inputs. It logs failed
reconstructions instead of replacing them. Feature caches already exist from E1,
so these timings should not be compared directly with cold E1 extraction times.
The Temple control has GT cameras; the car uses the same pseudo-reference as E1.

## E3 — number of views

Run `code/experiments/run_views.py`. All methods receive the same uniformly
spaced subset at each count. Car conditions are 5/10/20/30 views from the existing
60; Temple conditions are 5/10/16 because only 16 source images exist. Subsets
are selected independently, not falsely described as nested. Byte-identical E1
conditions reuse their logged runs with explicit provenance. Failed conditions
retain zero registration and zero pose AUC in the view-count plot; unavailable
ATE is not fabricated as zero.

For partial camera sets the benchmark still computes valid relative-pose AUC.
If fewer than three non-collinear cameras are shared, global Sim(3)/ATE remains
undefined rather than forcing a zero error or discarding all relative evidence.

## E4 — detector ablation

`run_ablation.py --experiment E4` tests SIFT, RootSIFT and ORB on the fixed car
and fountain images. It changes detector/descriptor normalization only; matcher
thresholds, initialization and BA settings stay fixed. ORB uses its appropriate
Hamming distance. Failed initialization is reported, not parameter-tuned away.

## E5 — intrinsics, public-scene substitution approved by user

`run_intrinsics.py` compares approximate K, supplied calibrated K and approximate
K with shared-focal global BA on Temple. The user confirmed no car checkerboard
data and selected this calibrated public-scene comparison. Approximate K uses
f=1.2*max(side) and a centered principal point. Known K is Middlebury calibration,
not a checkerboard calibration performed by this project.

Optional focal BA uses a single bounded log focal multiplier, a robust scalar
projection-fit initialization and the existing pose/point/scale-gauge objective.
It runs only in global BA after at least four cameras register; principal points
and distortion stay fixed. Default fixed-intrinsics BA is unchanged. Synthetic
known-camera tests recover a 25% focal error; the actual real-scene result is
retained even when joint self-calibration does not recover the true K. This is
an important limitation, not evidence that intrinsics can always be estimated.

## E6 — matching sensitivity

`run_ablation.py --experiment E6` varies the ratio threshold (0.7, 0.8, 0.9)
and Sampson/RANSAC threshold (0.75, 1.5, 3px), one factor at a time around the
frozen relaxed baseline. It does not retroactively replace E1 with the most
favorable setting. These are development sensitivity results on the same car.

## E7 — failure cases

`run_failures.py` compares both methods on the same glossy car, five-view
car subset, deliberately blurred calibrated Temple images (Gaussian radius 4px),
and the official 16-view low-texture Dino scene. Identical E1/E3 conditions reuse
their original runs with provenance. Blur is explicitly an artificial
intervention. Dino was not used for scene-specific parameter tuning. Failed
initializations and partial reconstructions remain in the tables and plots.

## M4 — real fountain-P11 GT mesh

`run_mesh_study.py --experiment M4` downloads the original EPFL triangle model,
camera files, P-matrices and bounding files into ignored `data/strecha/`, recording
URLs and SHA-256 hashes. The original 26-million-face mesh is not redistributed.
Strecha camera-to-world rotations and camera centers are converted to OpenCV
world-to-camera R/t and independently checked against every supplied P-matrix.

Meshing uses our frozen E1 fountain reconstruction, with no GT geometry or
cameras injected. Evaluation fits camera-center Sim(3), crops to the GT mesh's
axis-aligned bounding box, and samples 200,000 surface points per direction.
Accuracy/completeness and F-scores at 0.5%, 1%, 2% of the GT diagonal are saved.
Error colors and the GT preview use the same sampled reference surface. This
sample-based evaluation differs from the official EPFL visibility-aware scorer.
Completeness uses the whole GT scan, including regions that may be unseen in
input photos or removed by the sparse reconstruction ROI; no visibility mask
is applied. A low score is retained rather than selecting a favorable GT region.
Surface sampling processes face areas in bounded chunks to fit the original GT
without simplifying it or allocating all triangle coordinates at once.

## M5 — mesh source comparison

`run_mesh_study.py --experiment M5` reuses the exact M4 mesh and adds COLMAP poses
with the same Depth Anything V2 Small backend, filtering and TSDF parameters.
Camera, landmark alignment and ROI effects remain coupled; this is not a pose-only
causal comparison.

All eleven E1 fountain views are the fixed input set. Three deterministic view
names are excluded from filtering/fusion, but participate in camera estimation;
These are fusion-held-out rendering
diagnostics, not strictly unseen-image tests. PSNR/SSIM are masked to rendered
geometry and accompanied by coverage.
GT-aligned renders use the same native GT camera for a consistent visual view.

Each stage enforces completion/addressing of its predecessor in the ledger.
Raw failed attempts are retained; no favorable threshold replaces the frozen E1.

## Current two-method evidence

The active UI uses `validation/experiments_20261119_classical`, derived from the
retained 18 November measurements. Reconstruction scores are unchanged; removed
method rows are excluded and comparison plots are regenerated. Source file hashes
and derivation scope are recorded. Historical mixed-method evidence remains an
archive and is not served by the active UI. This is a scope projection, not a new
benchmark run.

```bash
.venv/bin/python code/experiments/classical_evidence.py
```

Fresh experiment output roots must have a protocol matching the current config.
Use a new `--out` when changing an old frozen protocol; do not overwrite its
measured attempts.
