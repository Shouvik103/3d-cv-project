# AGENT PIPELINE: From-Scratch Incremental SfM vs. CPU COLMAP

Course: 3D Computer Vision (IIIT Dharwad). Deliverable: `GroupX_3dcv_project.zip`.
Hard deadline: submit on **21 Nov 2026** (the brief says "before 22nd"). Target finishing on 20 Nov.

**Current scope — 19 November 2026:** Our SfM and CPU COLMAP are the two camera-reconstruction methods. Both support SGBM or optional Depth Anything V2 Small depth followed by TSDF. See [the current implementation and interface plan](docs/ACTIVE_PIPELINE.md). Earlier learned multi-view inference has been removed from implementation and deliverables. Historical records are archived, not active results.

**Contents**
- Section 0A: Source material (verbatim assignment brief, course context, why this project, data flow)
- Sections 1-18: project definition, rules, setup, layout, data, pipeline stages, baselines, evaluation, experiments, demo, tests, report, packaging, schedule, human tasks, risks, definition of done
- Appendix A: Math reference (every formula needed to implement and evaluate)
- Appendix B: Default configs (YAML)
- Appendix C: Code skeletons (data structures, BA, helpers, calibration, dense step)
- Appendix D: Synthetic test generator spec
- Appendix E: Local experiment execution
- Appendix F: README.md template
- Appendix G: Report skeleton (LaTeX), draft text, figure/table plan, reference list
- Appendix H: requirements.txt, run_all.sh, .gitignore
- Appendix I: Team workflow and interface contracts
- Appendix J: Troubleshooting FAQ
- Appendix K: Stretch: Nerfstudio/3DGS export
- Appendix L: Glossary and open items for humans

---

## 0. How to use this document

You are an autonomous coding agent working with a 4-person student team. Read the whole file first, then execute the stages in order. Each stage has: **goal, files, spec, acceptance test**. Do not move to the next stage until the acceptance test passes. If a stage fails twice after honest debugging, use the fallback in Section 17, log it in `NOTES.md`, and continue.

Primary machine: MacBook Air M4 (Apple Silicon, no CUDA). Camera estimation and SGBM run on CPU. Optional monocular depth can use MPS; the optional splat stretch goal remains separate.

---

## 0A. Source material and context

### 0A.1 Assignment brief (verbatim from the course)

```text
Package all your project files into a single .zip archive named GroupX_3dcv_project and upload it before 22nd November 2026
Group Rules (Max 4 Members)
* One Submission Per Group: Only one team member needs to upload the .zip file.
* Member Contributions: Include a brief Author Contributions table or paragraph at the end of your report stating what each member worked on (e.g., data preparation, model implementation, 3D visualization, report writing).
What to Include in the .zip File
Organize your archive cleanly with the following structure:
GroupX_3dcv_project/
├── report.pdf # Project report formatted in standard IEEE double-column style
├── demo.mp4 # 1–2 minute video showing your 3D outputs/renders
├── README.md # Simple steps to install dependencies and run your code
├── requirements.txt # Python packages needed (e.g., torch, open3d, numpy)
├── code/ # All source code (training, evaluation, and model files)
└── sample_data/ # 1–2 small test inputs so we can run a quick demo

Project Report (report.pdf) — IEEE Format
Format your report using the standard IEEE Conference double-column template (LaTeX or Word, 4–6 pages) and include these required sections in order:
* Title & Author Block: Project title, followed by all group members' names, roll numbers/IDs, and emails.
* I. Introduction: Brief background on the 3D computer vision topic and why it matters.
* II. Problem Statement: Clear definition of the exact problem your project solves, including inputs and expected 3D outputs.
* III. Literature Review: Summary of key existing methods/papers related to your task and their limitations.
* IV. Objectives: Bulleted list of the specific technical goals your team set out to achieve.
* V. Methodology: Your proposed pipeline, model architecture, 3D representations used (e.g., point clouds, meshes, voxels, NeRF/3DGS), loss functions, and datasets.
* VI. Results and Discussion: Quantitative metrics (tables/graphs), visual comparisons of your 3D outputs, ablation/error analysis, and failure cases.
* VII. Conclusion: Summary of your findings, current limitations, and potential future improvements.
* References: Properly numbered IEEE-style citations ([1], [2], etc.) matching all in-text citations.
Key Technical Rules
* Keep it under 200 MB: Do not put large datasets or heavy model weights inside the .zip. Upload large weights to Google Drive or Hugging Face (set to "Anyone with the link can view") and paste the link in README.md.
* No hardcoded paths: Use relative paths in your code so it runs on any computer without throwing file-not-found errors.
* Credit outside code: If you built on an existing repository (e.g., Nerfstudio, 3D Gaussian Splatting, PointNet++), cite it in your report and note what you modified in README.md.
Quick Check Before Submitting
* Unzip your archive in a brand-new folder and run your code on sample_data/ using the steps in README.md.
* Check that report.pdf strictly follows the IEEE two-column layout and has all 9 required elements above.
* Verify that demo.mp4 plays properly and all group members are tagged on the portal.
```

The "9 required elements" are: Title and author block, I Introduction, II Problem Statement, III Literature Review, IV Objectives, V Methodology, VI Results and Discussion, VII Conclusion, References.

### 0A.2 Course context (from the attached syllabus)

- **Course:** 3D Computer Vision, IIIT Dharwad (Institute of National Importance). Faculty: Dr. Shirshendu Layek. Credits: 3-0-0-4-4.
- **Prerequisites:** linear algebra, probability, programming, machine learning.
- **Course objectives:** (1) present core challenges of 3D vision; (2) explain essential principles and methods; (3) equip students to build solutions for moderately complex 3D vision tasks; (4) build comprehension of foundational methods in vision research.
- **Course learning outcomes:**
  - CO1: model camera systems; apply feature detection and stereo vision using OpenCV
  - CO2: use multi-view geometry for camera calibration and stratification with PyTorch3D
  - CO3: build 3D representations (point clouds, meshes) and apply reconstruction techniques
  - CO4: execute a complete 3D vision project integrating theory with OpenCV and PyTorch3D tools
  - CO5: execute an end-to-end 3D vision project combining geometric principles and deep architectures
- **Syllabus:**
  - Unit I: Camera Geometry, Calibration, Feature Tracking (projective geometry, image formation and camera model, feature detection and correspondence, stereo foundations)
  - Unit II: Multi-View Geometry and SfM (two-view and epipolar geometry, epipolar stratification, 3D data representations, SfM, dense reconstruction, scaling)
  - Unit III: Deep Learning on 3D Representations (point clouds, volumetric and mesh-based learning, 3D detection and scene understanding)
  - Unit IV: Implicit Representations, Neural Rendering, Advanced Pipelines (NeRF, end-to-end 3D deep pipelines, project presentations)
- **Books:** Szeliski, *Computer Vision: Algorithms and Applications*; Forsyth & Ponce, *Computer Vision: A Modern Approach*; Haralick & Shapiro, *Computer and Robot Vision*; Hartley & Zisserman, *Multiple View Geometry in Computer Vision*; Horn, *Robot Vision*.
- **Assessment:** assignments 15% (3), quizzes 10% (2), **project 25%**, mid-term 25%, end-term 25%.
- **Comparable courses named in the syllabus** (useful to cite as context in the Introduction, not as references): MIT 6.8300/6.8301, Oxford C19 (VGG), UT Austin CS 395T (3D vision), ETH Zürich 252-0579-00L 3D Vision, UVA CS 6501.

### 0A.3 Team and hardware

- Up to 4 members. Machine: MacBook Air M4 (Apple Silicon, no CUDA). Free Colab (T4 GPU, session limits, can disconnect).
- Requirement from the team: the project must be **substantial (not underwhelming) but not hardware-intensive**.
- Unknown at time of writing (ask humans, see Appendix L): group number X, member names/IDs/emails, phone model for capture, Mac RAM.

### 0A.4 Why this project (options considered)

| # | Option | Verdict |
|---|---|---|
| 1 | **Own incremental SfM vs. COLMAP** (transparent implementation vs. mature classical baseline) | **Chosen.** Mostly runs on Mac CPU; writing the geometry is substantive and defensible; fails gracefully; covers Units I, II, III |
| 2 | NeRF vs. 3DGS under sparse views (Nerfstudio) | Needs CUDA (Colab only); mostly running existing code, so ablations carry the grade. Kept as optional stretch (Appendix K) |
| 3 | Robust point cloud learning (PointNet/DGCNN on ModelNet40) | Feasible but visually weak; Unit III only |
| 4 | Video to mesh via monocular depth + TSDF fusion | Feasible; less geometry written by the team |
| 5 | Differentiable mesh fitting with PyTorch3D | Install is awkward on Mac; narrower |

Fallback if option 1 collapses badly: switch to option 2 or 3 using the same capture data. Decide by the end of Day 3 (Section 15).

### 0A.5 Mapping to course outcomes

| Outcome | Where it appears in this project |
|---|---|
| CO1 | camera model and intrinsics, checkerboard calibration, SIFT/ORB features, matching, stereo rectification and SGBM (dense step) |
| CO2 | epipolar geometry, F/E estimation, pose recovery, triangulation, PnP, BA. Note: the brief does not mandate PyTorch3D; optional PyTorch3D use is noted in Appendix C.8 |
| CO3 | sparse point clouds, dense point clouds, Poisson mesh, reconstruction evaluation |
| CO4 | complete project integrating OpenCV, SciPy, Open3D (and optionally PyTorch3D) |
| CO5 | complete geometric pipeline with optional learned monocular depth and neural rendering from our poses |

### 0A.6 What graders will most likely check (inferred from the brief; not an official rubric)

- Format compliance: zip name, folder structure, IEEE two-column report with all 9 elements and 4-6 pages, 1-2 min demo, README that works.
- Reproducibility: unzip, install, run on `sample_data/` without path errors, size under 200 MB.
- Technical depth: clear problem, correct methodology, quantitative results, ablations, failure cases, honest limitations.
- Attribution: outside code credited in report and README; author contributions at the end.

### 0A.7 End-to-end data flow

```text
photos (JPG/HEIC) ──prepare_data──► resized, EXIF-oriented images + intrinsics.yaml
        │                                   │
        │                        ┌──────────┴───────────────────────────────┐
        │                        ▼                                          ▼
        │            OUR SfM (code/sfm)                           BASELINES
        │   SIFT → match → RANSAC F → init pair          COLMAP (pycolmap/CLI, Mac CPU)
        │   → tracks → PnP register → triangulate             │
        │   → local/global BA → filter                            │
        │                        │                                │
        │                        ▼                                ▼
        │                 poses.json + points.ply  ◄── convert_to_common.py
        │                        │
        │                        ▼
        │        eval (Umeyama Sim(3) align → ATE/RPE/AUC/reproj/Chamfer/F-score/runtime)
        │                        │
        │                        ▼
        └────────► results/*.json → make_tables.py → report/tables/*.tex
                                 │
                                 └─► viz (orbit frames, growth animation) → demo.mp4
                                          report/main.tex (Overleaf) → report.pdf
                                          package → GroupX_3dcv_project.zip
```

---

## 1. Project at a glance

**Title (working):** *Incremental Structure from Motion: A From-Scratch Pipeline Benchmarked Against CPU COLMAP*

**Problem statement:**
- **Input:** an unordered set of N RGB photos (N = 3 to 200; about 60 suitable views recommended) of a static scene, plus approximate or calibrated intrinsics.
- **Output:** camera poses (R, t) for each image, a sparse colored 3D point cloud, and optionally a dense point cloud or mesh.

**Core contributions:**
1. Our own incremental SfM: feature matching, RANSAC two-view geometry, triangulation, PnP, and sparse bundle adjustment. The geometric core is written by us. OpenCV supplies SIFT/ORB, descriptor search, homography detection, PnP and image I/O; SciPy supplies nonlinear optimization. The mathematical core and this API boundary are described in docs/ACTIVE_PIPELINE.md.
2. A controlled benchmark of **ours vs. CPU COLMAP** on pose accuracy, reconstruction quality, runtime and robustness.
3. Ablations (detector, BA, number of views, intrinsics) and a failure-case study.

**Syllabus mapping (state this in the report):**
- Unit I: camera model, calibration, SIFT/ORB, stereo geometry
- Unit II: epipolar geometry, SfM, triangulation, BA, dense reconstruction
- Unit III: optional learned monocular depth aligned to classical sparse structure
- Unit IV (optional): splat/NeRF fly-through from our poses

---

## 2. Hard constraints from the assignment

- Zip name `GroupX_3dcv_project` (replace X with the group number). One upload per group, under **200 MB**.
- Exact structure:
```
GroupX_3dcv_project/
├── report.pdf        # IEEE double-column, 4-6 pages
├── demo.mp4          # 1-2 min, shows 3D outputs/renders
├── README.md         # install + run steps, links to large weights, third-party credits
├── requirements.txt
├── code/             # all source (pipeline, baselines, evaluation)
└── sample_data/      # 1-2 small inputs so graders can run a quick demo
```
- No hardcoded absolute paths. Use `pathlib` relative to the file location.
- Credit all outside code in the report and list what we modified in the README.
- Report sections in order: Title/Authors, I Introduction, II Problem Statement, III Literature Review, IV Objectives, V Methodology, VI Results and Discussion, VII Conclusion, References. Author Contributions goes at the very end.
- Model weights are never inside the zip.

---

## 3. Operating rules for the agent

1. **Never fabricate results.** Every number, table and figure in the report must come from a logged run. Every experiment writes a JSON to `results/` containing config, seed, git hash or timestamp, and metrics. Tables are generated from those JSONs by script.
2. Work in small commits. Run unit tests after every module.
3. Seed everything (`numpy`, `random`, `torch`). Record seeds in the result JSONs.
4. Only use relative paths. Configuration lives in `configs/*.yaml`.
5. Never run long jobs in the foreground without progress logging (`tqdm` plus a log file).
6. If a download is blocked or a library API differs from this document, check the library's current README or docs and adapt. Record the change in `NOTES.md`.
7. Tasks only humans can do are listed in Section 16. When you reach one, stop and ask clearly instead of guessing.
8. Keep a running `NOTES.md` with decisions, failures, and anything needed for the report (parameter values, versions, timings).

---

## 4. Environment setup (Mac, Apple Silicon)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy opencv-python open3d matplotlib tqdm pyyaml pandas pillow imageio imageio-ffmpeg pytest exifread pycolmap
pip install torch torchvision          # optional learned monocular depth / MPS
brew install ffmpeg                    # for demo.mp4 (fallback: imageio-ffmpeg)
```

Notes:
- Use Python 3.11 for the widest wheel support (Open3D, pycolmap). If `pycolmap` has no wheel for the platform, use `brew install colmap` and call the CLI via `subprocess`.
- COLMAP on Mac runs CPU-only. That is fine: we only need feature extraction, matching and sparse mapping. **Do not attempt COLMAP dense stereo (needs CUDA).**
- Pin the final versions in `requirements.txt` using only packages actually imported by `code/` (not a full `pip freeze`).
- Log versions of Python, OpenCV, pycolmap, Open3D and torch into `NOTES.md` (they go in the report).

---

## 5. Repository layout (work in `GroupX_3dcv_project/`)

```
GroupX_3dcv_project/
├── README.md
├── requirements.txt
├── NOTES.md                      # remove or keep out of final zip if noisy
├── configs/
│   ├── sfm_default.yaml
│   └── eval.yaml
├── code/
│   ├── sfm/
│   │   ├── features.py           # SIFT/ORB extraction + cache
│   │   ├── matching.py           # ratio test, mutual check, geometric verification
│   │   ├── geometry.py           # 8-pt F, RANSAC, E decomposition, DLT triangulation, PnP helpers
│   │   ├── tracks.py             # union-find feature tracks
│   │   ├── reconstruction.py     # data structures (cameras, points, observations)
│   │   ├── incremental.py        # initialization + registration loop
│   │   ├── bundle_adjust.py      # sparse BA with scipy
│   │   └── io_utils.py           # PLY, poses.json, intrinsics loading
│   ├── baselines/
│   │   ├── run_colmap.py         # pycolmap or CLI wrapper -> poses.json + ply
│   │   └── convert_to_common.py # common COLMAP pose / point export
│   │   └── convert_to_common.py  # npz/colmap -> common poses.json
│   ├── eval/
│   │   ├── align.py              # Umeyama Sim(3)
│   │   ├── metrics.py            # ATE, RPE, AUC, reprojection, Chamfer, F-score
│   │   ├── run_eval.py           # compares methods on a scene -> results JSON
│   │   └── make_tables.py        # JSON -> CSV + LaTeX tables
│   ├── viz/
│   │   ├── render_orbit.py       # orbit frames of a point cloud + camera frustums
│   │   ├── plot_matches.py       # matches / epipolar line figures
│   │   └── make_demo_frames.py
│   ├── calibrate_camera.py       # checkerboard calibration (Unit I)
│   ├── run_sfm.py                # CLI entry point
│   └── run_all.sh
├── notebooks/
│   └── # no GPU reconstruction notebook required
├── tests/                        # pytest, synthetic-data tests
├── sample_data/                  # small demo inputs (see Section 14)
├── data/                         # NOT zipped: raw photos, public datasets
├── results/                      # NOT zipped (except tiny summary tables)
└── report/                       # LaTeX source, figures
```

**Common pose format** (`poses.json`), shared by every method so evaluation is method-agnostic:
```json
{
  "image_names": ["IMG_0001.jpg", ...],
  "K": {"IMG_0001.jpg": [[fx,0,cx],[0,fy,cy],[0,0,1]]},
  "R": {"IMG_0001.jpg": [[...3x3...]]},     // world-to-camera, OpenCV convention
  "t": {"IMG_0001.jpg": [tx,ty,tz]},        // x_cam = R X + t, camera center C = -R^T t
  "registered": ["IMG_0001.jpg", ...],
  "image_size": {"IMG_0001.jpg": [W,H]}
}
```
Convention rule: everything internal is OpenCV (x right, y down, z forward), world-to-camera. Convert COLMAP (`cam_from_world`) into this on import and test the conversion.

---

## 6. Data

### 6.1 Self-captured scenes (human task, see Section 16)
Capture protocol to give the team:
- Use one phone, **1x main lens, no zoom, no portrait/HDR/Live mode, lock focus and exposure** if possible. Do not change the lens between shots.
- Avoid motion blur. Overlap 60-80% between neighbors. Move around the subject in steps of about 10-15 degrees.
- Scenes (aim for 3):
  - **S1 (easy):** richly textured object on a table (books, plants, a sculpture). 30-40 photos in a full **360-degree ring**. The full loop is deliberate: without loop closure it reveals drift, which is great analysis material.
  - **S2 (medium):** outdoor corner of a building or a campus feature, 25-35 photos on an arc.
  - **S3 (hard, failure study):** low-texture or repetitive subject (plain wall with a poster, window grid, shiny object) or very sparse views (8-10 photos).
- Optionally 15-20 photos of a printed checkerboard (e.g., 9x6 inner corners, known square size) from varied angles for calibration.

Preprocessing (`code/sfm/io_utils.py` plus a `code/prepare_data.py` script):
- Resize so the long side is at most 1600 px (store a copy under `data/<scene>/images_1600/`). Preserve aspect ratio.
- Read EXIF `FocalLengthIn35mmFilm` (via `exifread` or PIL). Initial intrinsics: `f_px = f35 / 36 * max(W, H)`, principal point at image center, zero distortion. If EXIF is missing, use `f_px = 1.2 * max(W, H)`.
- If a checkerboard calibration exists, use calibrated K (and undistort) as the "calibrated" setting. Otherwise EXIF K. Keep both for the intrinsics ablation.
- Scale K when resizing images.

### 6.2 Public scene with ground-truth poses (needed for real accuracy numbers)
Self-captured scenes have no true ground truth, so COLMAP is only a *pseudo*-ground-truth there. Add one public scene with true camera poses:
- **Preferred:** Middlebury Multi-View Stereo "templeSparseRing" and/or "dinoSparseRing" (16 views each, ground-truth K, R, t in the dataset's parameter text file; verify the file format on the dataset page). Small, fast, and also a good low-texture stress test (dino).
- **Alternative:** EPFL/Strecha `fountain-P11` or `Herz-Jesu-P8` (GT cameras provided). Verify the camera-file rotation convention: if ATE vs. COLMAP is huge after Sim(3) alignment, try the transposed rotation.
- If the workspace cannot download, ask the humans to download and place it in `data/`. Write a loader that converts GT into the common `poses.json`.
- Always validate a GT loader: project known-good COLMAP points with GT cameras, or compare GT vs. COLMAP poses after Sim(3) alignment. They should agree closely.

### 6.3 What counts as ground truth in the report
- Public scene: GT poses are the reference, so report true errors.
- Self-captured scenes: COLMAP is a pseudo-reference. Say so explicitly. Metrics there measure *agreement with COLMAP*, not truth.

---

## 7. Our SfM pipeline (the core: implement all of this ourselves)

### Stage 1. Feature extraction (`features.py`)
- `extract_features(img_gray, detector="sift"|"orb", n_features=8000) -> (keypoints[N,2], descriptors[N,D])`.
- SIFT via `cv2.SIFT_create(nfeatures=n)`. Optionally apply **RootSIFT** (L1-normalize, then sqrt). Make it a config flag; it is a cheap ablation.
- ORB via `cv2.ORB_create(nfeatures=n)`.
- Cache to `data/<scene>/cache/features_<detector>.npz`.
- Accept: ≥1500 keypoints on typical textured images, deterministic across runs.

### Stage 2. Matching and geometric verification (`matching.py`, `geometry.py`)
- Pair selection: exhaustive for N ≤ 40 (≤ 780 pairs).
- Descriptor matching: kNN (k=2) with Lowe ratio test (default 0.8), plus mutual-nearest-neighbor check. Use `cv2.BFMatcher` (L2 for SIFT, Hamming for ORB). If it is too slow, cap features at 4000 or use FLANN.
- **Implement ourselves** in `geometry.py`:
  - `normalize_points(x) -> (x_n, T)` (Hartley normalization)
  - `fundamental_8pt(x1, x2) -> F` (normalized 8-point, enforce rank 2 via SVD, denormalize)
  - `sampson_distance(F, x1, x2) -> d[N]`
  - `ransac_fundamental(x1, x2, thresh=1.0, iters=2000, confidence=0.999, seed=0) -> (F, inlier_mask)` with **adaptive iteration count**
- Verify against `cv2.findFundamentalMat(..., cv2.FM_RANSAC)` in a test: inlier sets should overlap strongly.
- Keep pairs with ≥ 30 inliers (config). Store verified matches per pair in `data/<scene>/cache/matches.npz`.
- Also estimate a homography per pair with `cv2.findHomography` (RANSAC) and record `H_inliers / F_inliers`. It feeds the initialization pair choice (Stage 3) and is great failure-analysis material (pure rotation or planar scenes).

### Stage 3. Two-view initialization (`incremental.py`, `geometry.py`)
- Choose the initial pair by score: many F-inliers, **low** H/F inlier ratio (< 0.7), and sufficient triangulation angle (median ≥ 2-3 degrees). Evaluate the top 10 candidate pairs by inlier count and pick the best score.
- `essential_from_F(F, K1, K2) = K2^T F K1`
- `decompose_essential(E) -> four (R, t)` candidates via SVD (both W and W^T, both signs of t).
- **Cheirality check:** triangulate with each candidate and keep the one with the most points in front of both cameras. Implement this ourselves (cross-check with `cv2.recoverPose` in tests).
- `triangulate_dlt(P_list, x_list) -> X` for **n views** (homogeneous linear system via SVD). Use normalized coordinates for conditioning.
- Filter new points: reprojection error < 2 px in both views, positive depth, triangulation angle ≥ 1.5 degrees.
- Set camera 0 = identity pose, camera 1 = (R, t) with ||t|| = 1 (fix scale). Run BA on the initial pair.

### Stage 4. Tracks (`tracks.py`)
- Union-find over verified matches `(image_id, kp_id)` to form tracks.
- Reject tracks that contain two different keypoints in the same image.
- Record the track length distribution (it goes in the report).

### Stage 5. Incremental registration (`incremental.py`)
Loop until no image can be registered:
1. Select the unregistered image with the most 2D-3D correspondences (keypoints belonging to tracks that already have a 3D point).
2. Estimate pose via `cv2.solvePnPRansac(X, x, K, None, reprojectionError=4.0, iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)` followed by `cv2.solvePnPRefineLM` on the inliers. Require ≥ 20 inliers, otherwise skip the image for now and retry later.
   - **Also implement DLT-PnP ourselves** in `geometry.py` (6+ points, SVD) and use it in a unit test to validate `solvePnPRansac`. Using OpenCV's RANSAC wrapper in the main loop is allowed because the PnP model is still ours to explain in the report; state this honestly.
3. Triangulate new tracks that now have ≥ 2 registered views (n-view DLT, same filters as Stage 3).
4. **Local BA** (Stage 6) on the newly registered camera and its 4-5 most connected neighbors.
5. Run **global BA** every `k` registrations (default 5) and once at the end, then filter points with reprojection error > 3 px and re-run BA once.

Log per step: registered count, number of points, mean reprojection error (px), runtime.

### Stage 6. Bundle adjustment (`bundle_adjust.py`)
- Parameters: per camera 6 DOF (Rodrigues vector 3 + translation 3), per point 3. Optionally a shared focal length (flag `refine_focal`).
- Residuals: 2D reprojection residuals (2 per observation).
- Solver: `scipy.optimize.least_squares` with `jac_sparsity` (sparse structure built from the camera-point observation graph), `method="trf"`, `tr_solver="lsmr"`, `x_scale="jac"`, `loss="huber"`, `f_scale=1.0` to `2.0`, `max_nfev` ~ 50-100.
- Gauge fixing: keep camera 0 fixed (exclude it from the parameter vector). Scale is fixed by the initial pair.
- Speed: only include points with track length ≥ 2 observed by ≥ 2 registered cameras. For global BA, cap at about 10k points (keep the longest tracks first).
- Accept: on the synthetic test, BA reduces RMS reprojection error to the noise level and recovers camera poses within tolerance (see Section 12).

### Stage 7. Export (`io_utils.py`)
- `poses.json` (common format), `points.ply` (XYZ + RGB sampled from the first observing image), `cameras.ply` or a frustum lineset for visualization.
- Statistical outlier removal on the cloud for display (Open3D), keeping the raw cloud for evaluation.

### Stage 8. Optional dense step (stretch, gated; only if the core is done by end of Day 5)
Cheap dense reconstruction without CUDA:
- Select K = 3 to 5 neighbor image pairs with good baseline. Rectify with `cv2.stereoRectify` (using our R, t, K), compute disparity via `cv2.StereoSGBM_create`, reproject to 3D with `cv2.reprojectImageTo3D`, transform to world frame.
- Merge into one Open3D cloud, downsample, estimate normals, and run Poisson reconstruction (depth 8) for a mesh. Trim low-density vertices.
- This covers "stereo vision" and "dense reconstruction" from the syllabus. Mark it clearly as a qualitative result. If results are poor, report that honestly in the failure cases.

### CLI (`code/run_sfm.py`)
```bash
python code/run_sfm.py --images sample_data/scene1/images --out outputs/scene1_ours \
    --config configs/sfm_default.yaml [--detector sift|orb] [--no_ba] [--max_images 20]
```
It must work from the repo root on any machine, using relative paths only.

---

## 8. Baselines

### 8.1 COLMAP (`baselines/run_colmap.py`)
- Run on the **same resized images** and the same camera assumption: single camera, `SIMPLE_RADIAL` (or `PINHOLE` with our K fixed) so that the comparison is fair. Record which was used.
- Pipeline: feature extraction (SIFT, CPU), exhaustive matching, incremental mapper. Use `pycolmap` (`pycolmap.extract_features`, `match_exhaustive`, `incremental_mapping`) or the CLI. Check the installed pycolmap API version because it has changed over releases.
- If multiple models are produced, keep the one with the most registered images.
- Convert to the common `poses.json` plus `points.ply`. Measure runtime.

### 8.2 Common dense reconstruction

Both methods export independent cameras, points and observed tracks. The mesh
add-on consumes those common exports. SGBM estimates calibrated stereo depth;
optional monocular inverse depth is aligned to that method's sparse landmarks.
Foreground, ROI and multi-view consistency gates precede TSDF fusion. See
MESH_ADDON.md and docs/ACTIVE_PIPELINE.md for current presets and diagnostics.


## 9. Evaluation (`code/eval/`)

All metrics are computed on the common pose format and PLY files by `run_eval.py`. Output: `results/<scene>/<method>_<tag>.json` and CSV tables.

**Alignment:** Umeyama Sim(3) alignment of estimated camera centers to the reference (`align.py`). Report both the similarity transform and residuals. Only compare commonly registered images and report the registration rate separately.

**Metrics (`metrics.py`):**
| Metric | Definition |
|---|---|
| Registration rate | registered / total images |
| ATE (aligned) | RMSE of camera-center error after Sim(3); report in scene units and as % of camera-trajectory extent |
| Relative rotation error (deg) | per image pair: `arccos((tr(R_est R_ref^T) - 1)/2)` on relative poses |
| Relative translation-direction error (deg) | angle between estimated and reference baseline directions per pair |
| Pose AUC@5/10/30 | area under the cumulative curve of `max(rot_err, trans_err)` over all pairs |
| Reprojection error (px) | mean/median on original-resolution images, our SfM and COLMAP only |
| # 3D points, mean track length | for our SfM and COLMAP |
| Chamfer distance, F-score@τ | point cloud vs. reference cloud after alignment; τ = 1% of scene extent. Use sampled/limited points for speed |
| Runtime, peak memory | wall-clock per method on the recorded CPU hardware |

**References:**
- Public GT scene: GT poses (and GT cloud if available).
- Self-captured scenes: COLMAP result as pseudo-reference (state this in tables and captions).

Write `make_tables.py` to emit LaTeX tables (`report/tables/*.tex`) directly from the JSONs.

---

## 10. Experiments to run (prioritized; cut from the bottom if time runs out)

| # | Experiment | Methods | Scenes | Priority |
|---|---|---|---|---|
| E1 | Main comparison: all metrics | Ours, COLMAP | S1, S2, public-GT | **Must** |
| E2 | BA ablation: no BA vs. local only vs. local+global | Ours | S1 (loop shows drift) | **Must** |
| E3 | Number of views N ∈ {5, 10, 20, 30/40} | Ours, COLMAP | S1 (+ public GT) | **Must** |
| E4 | Detector ablation: SIFT vs. RootSIFT vs. ORB | Ours | S1, S2 | Should |
| E5 | Intrinsics: EXIF guess vs. checkerboard-calibrated vs. refined in BA | Ours | S1 | Should |
| E6 | Ratio threshold (0.7, 0.8, 0.9) and RANSAC threshold | Ours | S1 | Nice |
| E7 | Failure cases: texture-less / repetitive / sparse / blur | All | S3, dino | **Must** (qualitative + a few numbers) |
| E8 | Dense reconstruction/mesh qualitative | Ours | S1 | Stretch |
| E9 | Splat or NeRF fly-through from our poses (Colab) | Nerfstudio splatfacto | S1 | Stretch, hard time cap of half a day |

Failure-case discussion points to look for: homography-dominated pairs (planar scenes), too-small baselines, repeated structures causing wrong matches, drift around a closed loop, COLMAP splitting into multiple models.

Stretch E9 note: write `transforms.json` (Nerfstudio format) from our poses instead of running `ns-process-data` (which needs COLMAP). Keep resolution small. Only do this if E1-E3, E7 and the report draft are done.

---

## 11. Visualization and demo (`code/viz/`, `demo.mp4`)

- Figures for the report: keypoints and verified matches, epipolar lines on an image pair, pose trajectory plots (ours vs. reference, aligned), reprojection error histograms, point cloud screenshots side by side (ours / COLMAP), AUC curves, BA ablation drift plot, failure-case panels.
- Rendering on macOS: Open3D headless offscreen rendering can be unreliable. Prefer `o3d.visualization.Visualizer` with a visible window and `capture_screen_image` in a loop (rotate the view each frame), or fall back to a matplotlib scatter-based renderer. Test early.
- Orbit frames: render 150-240 frames at 1280x720, 30 fps. Draw camera frustums as a lineset.
- **Incremental growth animation:** save the reconstruction state after each registration step and render a frame per step. It visualizes how SfM works and fills time in the video.
- Demo storyboard (90-110 s total, under about 25 MB):
  1. 0-8 s: title and one-line problem
  2. 8-20 s: photo montage (inputs)
  3. 20-35 s: matches and epipolar geometry
  4. 35-55 s: incremental reconstruction growing, cameras appearing
  5. 55-85 s: side-by-side orbits: Ours | COLMAP with labels
  6. 85-100 s: key metrics table overlay
  7. 100-110 s: team names, optional splat fly-through
- Assemble: `ffmpeg -framerate 30 -i frames/%05d.png -c:v libx264 -pix_fmt yuv420p -crf 23 demo.mp4`. Add text via `cv2.putText` or `PIL` when making frames. Verify the file plays (`ffprobe demo.mp4`; duration 60-120 s).

---

## 12. Unit tests and acceptance criteria (`tests/`, run with `pytest`)

Create a **synthetic scene generator** first (`tests/synthetic.py`): random 3D points in a box, ~10 cameras on an arc looking at the center, known K, project with optional Gaussian pixel noise (σ = 0.5 px) and a small outlier fraction.

| Test | Pass criterion |
|---|---|
| `fundamental_8pt` on noise-free data | Sampson error < 1e-6 px |
| `ransac_fundamental` with 30% outliers | ≥ 90% of true inliers recovered |
| `decompose_essential` + cheirality | recovers true R within 0.1° (noise-free) and t direction within 0.1° |
| `triangulate_dlt` (n views) | point error < 1e-6 (noise-free); < 0.5% of scene size at σ=0.5 px |
| DLT-PnP vs. truth | rotation error < 0.5° at σ=0.5 px |
| Pose conversions (COLMAP, Middlebury) | round-trip tests with known transforms |
| Umeyama | recovers a random Sim(3) to 1e-8 |
| BA on noisy synthetic data (perturbed init) | final RMS reprojection error within 20% of noise σ; camera center errors reduced vs. the initialization |
| Full pipeline on the synthetic scene | all cameras registered, ATE after alignment < 1% of scene extent |
| Real data smoke test (sample_data) | pipeline finishes in < 5 min on the M4; registers ≥ 90% of images; median reprojection error < 1.5 px |

Do not start experiments (Section 10) until the synthetic tests pass.

---

## 13. Report (`report/`, output `report.pdf`)

- Template: IEEE conference double-column (`\documentclass[conference]{IEEEtran}`). 4-6 pages. Humans compile in Overleaf (a local TeX install is big). The agent writes `main.tex`, `refs.bib` (or `thebibliography`), tables and figures.
- Section order and budget (approx.):
  - Title and author block (names, roll numbers/IDs, emails: ask humans for these)
  - I. Introduction (0.4 page): why SfM and 3D reconstruction matter (AR, robotics, mapping, heritage)
  - II. Problem Statement (0.3 page): inputs, outputs, evaluation scope
  - III. Literature Review (0.8 page): classical SfM, COLMAP, bundle adjustment, optional monocular depth, neural rendering (NeRF/3DGS) context; **state limitations** of each
  - IV. Objectives (bulleted, 0.2 page)
  - V. Methodology (1.2 pages): pipeline diagram figure, math for each stage (8-point, E decomposition, DLT, PnP, BA cost with Huber), baselines and their configuration, datasets and capture protocol, metric definitions
  - VI. Results and Discussion (1.5-2 pages): tables (E1), figures (pose plots, point clouds side by side), ablations (E2-E6), failure cases (E7), runtime
  - VII. Conclusion (0.3 page): findings, limitations, future work (loop closure, learned matchers such as SuperPoint/LightGlue, dense MVS, 3DGS)
  - References: numbered IEEE style, every entry cited in text, and vice versa
  - Author Contributions: short table at the very end (who did data capture, SfM modules, baselines, evaluation, visualization/video, report). Ask the humans for the real split. Do not invent it.
- Suggested references (the agent must **verify bibliographic details** via search rather than rely on memory): Lowe (SIFT, IJCV 2004); Rublee et al. (ORB, ICCV 2011); Fischler & Bolles (RANSAC, 1981); Hartley & Zisserman, *Multiple View Geometry* (CUP); Nistér (five-point, TPAMI 2004); Triggs et al., "Bundle Adjustment: A Modern Synthesis"; Schönberger & Frahm (COLMAP, CVPR 2016); Wang et al. (DUSt3R, CVPR 2024); Umeyama (1991); Zhou et al. (Open3D, 2018); Szeliski, *Computer Vision: Algorithms and Applications*; dataset papers (Middlebury MVS: Seitz et al. 2006; Strecha et al. 2008 if used); Mildenhall et al. (NeRF) and Kerbl et al. (3DGS) if discussed.
- Every outside repo used (OpenCV, COLMAP/pycolmap, Open3D, SciPy, Nerfstudio if used) must be cited in the report.
- After compiling: check page count (4-6), two-column layout, and that all 9 required elements exist.

---

## 14. Packaging and final checks

`README.md` must contain:
1. Project summary (3-4 lines) and the folder structure.
2. Install steps (Python version, `pip install -r requirements.txt`, ffmpeg if needed).
3. Quick demo: exact command(s) on `sample_data/`, expected outputs, approximate runtime.
4. How to reproduce experiments (`code/run_all.sh`), and which optional stages need learned depth weights.
5. Links for large files (optional monocular depth weights are downloaded automatically from Hugging Face; any large dataset or precomputed outputs on Google Drive/HF with "Anyone with the link can view").
6. **Third-party code and what we modified** (table: library/repo, version, how used).
7. Team and contribution summary.

`sample_data/`:
- `scene_small/images/` with 10-12 downsized images (≤ 800 px long side, about 2-4 MB total), `intrinsics.yaml`, and precomputed COLMAP `poses.json` for that scene, so the demo runs with no GPU. Optionally a tiny second input (the public GT scene subset, if license permits redistribution; otherwise just describe how to download it).

Build the zip (macOS adds junk files; exclude them):
```bash
cd .. && zip -r GroupX_3dcv_project.zip GroupX_3dcv_project \
  -x "*.DS_Store" "*__pycache__*" "*.ipynb_checkpoints*" "*/.venv/*" "*/data/*" "*/results/*" "*/outputs/*" "*/.git/*"
du -sh GroupX_3dcv_project.zip   # must be well under 200 MB
```

**Fresh-machine test (mandatory):**
```bash
rm -rf /tmp/fresh && mkdir /tmp/fresh && cd /tmp/fresh
unzip /path/to/GroupX_3dcv_project.zip && cd GroupX_3dcv_project
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# run the README quick-demo command exactly as written
grep -rn "/Users/\|C:\\\\" code/ || echo "no hardcoded paths"
```

Final checklist:
- [ ] Folder names and file names match Section 2 exactly
- [ ] `report.pdf`: IEEE double-column, 4-6 pages, all 9 elements, authors with IDs and emails, contributions at the end, all citations resolve
- [ ] `demo.mp4`: plays, 60-120 s, shows our result, COLMAP and ideally the incremental growth
- [ ] Quick demo from README works on a fresh unzip
- [ ] No weights or datasets in the zip; total size < 200 MB
- [ ] No hardcoded paths; third-party credits in README and report
- [ ] All group members tagged on the portal (human task)

---

## 15. Schedule (today is Sat 14 Nov; submit by Sat 21 Nov at the latest)

| Day | Date | Goal | Gate to pass |
|---|---|---|---|
| 1 | Sat 14 | Environment, repo skeleton, synthetic generator and tests for geometry primitives. **Humans: capture S1 (+ S2/S3 and checkerboard if possible).** | primitives tests green |
| 2 | Sun 15 | Features, matching, own RANSAC F, initialization with cheirality, triangulation. Run COLMAP baseline on S1. Download public GT scene. | two-view recon on S1 looks sane |
| 3 | Mon 16 | Tracks, PnP registration loop, BA (local + global). | full SfM on S1 registers ≥ 90% images |
| 4 | Tue 17 | Common pose format, evaluation code. Run the two independent classical methods locally. | E1 table for S1 produced |
| 5 | Wed 18 | Experiments E1-E3, E7; E4/E5 if time. Optional dense step. | all "Must" experiments have result JSONs |
| 6 | Thu 19 | Figures, orbit renders, demo.mp4; start report writing in parallel. | demo video draft exists |
| 7 | Fri 20 | Finish report, compile in Overleaf, packaging, fresh-machine test. | checklist all ticked |
| 8 | Sat 21 | **Buffer + upload** (do not rely on this day). | submitted |

**Cut line:** if Day 3 ends without a working incremental pipeline, drop global BA (keep local), and drop E4-E6, E8, E9. If Day 5 ends late, drop E5-E6 and the dense step. Never cut the synthetic tests, E1, E2, E3, E7 or the fresh-machine test.

Suggested ownership (4 members): (A) data capture, calibration, COLMAP baseline and public dataset loader; (B) features, matching, two-view geometry, initialization; (C) tracks, PnP, bundle adjustment, incremental loop; (D) evaluation, visualization, mesh analysis and demo. Report sections and README shared by all.

---

## 16. Tasks only humans can do (the agent must stop and ask)

1. Capture the photos, and optionally the checkerboard set. Put them in `data/<scene>/images_raw/`.
2. Review local camera calibration, frame selection and the independent two-method comparison.
3. Provide names, roll numbers/IDs and emails for the author block, and the real contribution split.
4. Compile the LaTeX in Overleaf and return `report.pdf` if no local TeX exists.
5. Upload large optional files to Drive/HF if any, and paste the links.
6. Upload the zip to the portal and tag all members.

---

## 17. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Own BA is too slow or unstable | Cap points to the longest 3-5k tracks; local BA only; document the drift this causes (it becomes the E2 ablation) |
| Registration loop breaks on hard scenes | Lower PnP inlier requirement to 15, raise reprojection threshold to 6 px, relax Lowe ratio to 0.85; report remaining failures as limitations |
| pycolmap not installable on Mac | `brew install colmap` and use the CLI from `run_colmap.py` |
| Open3D rendering fails on macOS | matplotlib or software point rasterizer; or screen-capture the interactive viewer |
| Reconstruction exceeds available memory | bound view count/resolution, use serial CPU jobs and retain partial/failure records |
| Public GT dataset unreachable | use another GT scene from the list in Section 6.2; if none, use COLMAP-as-reference only and state this clearly |
| Scale or convention mismatches in evaluation | check conventions via unit tests; evaluate after Sim(3) alignment; if error is huge, try transposed R and verify with a known-good COLMAP result |
| Time overrun | follow the cut line in Section 15 |

---

## 18. Definition of done

The project is done when: the synthetic tests pass; our SfM reconstructs S1 and S2 from `sample_data` in a few minutes on the M4; E1, E2, E3 and E7 results exist as JSON with scripts that rebuild the tables; `report.pdf` and `demo.mp4` meet the format rules; the zip passes the fresh-machine test; and all numbers in the report trace back to logged runs.

---

# APPENDICES

## Appendix A. Math reference

Conventions: homogeneous image point `x = (u, v, 1)^T`; world point `X`; camera `x_cam = R X + t` (world-to-camera, OpenCV axes); camera center `C = -R^T t`; projection `x ~ K (R X + t)`, with `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`. Normalized coordinates: `x_n = K^{-1} x`. Two-view convention: `x2_cam = R x1_cam + t`, so `P1 = K1 [I|0]`, `P2 = K2 [R|t]`.

### A.1 Normalized 8-point algorithm (fundamental matrix)
1. Normalize each point set: translate the centroid to the origin and scale so the mean distance from the origin is `sqrt(2)`. This gives `T1`, `T2` and normalized points `x1', x2'`.
2. Epipolar constraint `x2'^T F' x1' = 0`. Each correspondence `(u1,v1)`, `(u2,v2)` gives the row
   `[u2 u1, u2 v1, u2, v2 u1, v2 v1, v2, u1, v1, 1]` acting on `vec(F')` (row-major).
3. Solve `A f = 0` by SVD: `f` = right singular vector of the smallest singular value; reshape to 3x3.
4. Enforce rank 2: `F' = U diag(s1, s2, 0) V^T`.
5. Denormalize: `F = T2^T F' T1`.

Sampson distance (first-order geometric error, squared pixels):
`d = (x2^T F x1)^2 / ( (F x1)_1^2 + (F x1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2 )`.
Inlier if `sqrt(d) < thresh` (default 1.0 px).

Adaptive RANSAC iterations: `N = log(1 - p) / log(1 - w^s)`, sample size `s = 8`, confidence `p = 0.999`, inlier ratio `w` = best so far. Cap at `ransac_iters`. After RANSAC, refit F on all inliers (8-point) and re-evaluate inliers once.

### A.2 Essential matrix and pose recovery
- `E = K2^T F K1` (optionally project to singular values (1, 1, 0)).
- SVD `E = U diag(1,1,0) V^T`; if `det(U) < 0` flip the sign of the last column of `U`; same for `V`.
- `W = [[0,-1,0],[1,0,0],[0,0,1]]`; `R_a = U W V^T`, `R_b = U W^T V^T`; `t = ± u3` (last column of `U`).
- Four candidates: `(R_a, +t)`, `(R_a, -t)`, `(R_b, +t)`, `(R_b, -t)`. Triangulate inliers with each. Choose the candidate with the largest number of points with positive depth in **both** cameras (cheirality). Ensure `det(R) = +1`.
- Two-view translation scale is arbitrary: fix `||t|| = 1`.

### A.3 n-view DLT triangulation
For each view `i` with normalized observation `(a_i, b_i)` and normalized projection `P_i = [R_i | t_i]` (rows `p1, p2, p3`): rows `a_i p3 - p1` and `b_i p3 - p2`. Stack into a `2n x 4` matrix `A`; `X_h` = right singular vector for the smallest singular value; `X = X_h[:3] / X_h[3]`.
- Reject if `|X_h[3]|` is tiny or depth ≤ 0 in any view.
- Triangulation angle between two views: `cos θ = (X - C1)·(X - C2) / (|X - C1| |X - C2|)`. For n views use the maximum pairwise angle.
- Reprojection error in pixels: project with `K`, compare with the original pixel observation.

### A.4 DLT PnP (used in tests and as an initializer)
Given ≥ 6 correspondences `X_j ↔ (a_j, b_j)` in normalized coordinates, unknown `P = [M | p4]` (3x4):
rows `[X_j^T, 0, -a_j X_j^T]` and `[0, X_j^T, -b_j X_j^T]` with homogeneous `X_j = (X, Y, Z, 1)`. Solve `A p = 0` by SVD (12 unknowns). Then `M = U S V^T`; `R = U V^T` (fix `det = +1`; if `det(M) < 0` negate `P` first); scale `λ = mean(S)`; `t = p4 / λ`. Refine with `cv2.solvePnPRefineLM`.

### A.5 Bundle adjustment objective
`min over {R_i, t_i}, {X_j}:  Σ_{(i,j) observed} ρ( || π(K_i, R_i, t_i, X_j) - x_ij ||^2 )`
with `π` the pixel projection, `ρ` the Huber loss. Parameters: camera `(rvec[3], t[3])`, point `X[3]`. Each residual (2 values) depends only on one camera (6 params) and one point (3 params), which yields the Jacobian sparsity pattern.
Rodrigues rotation of vector `p` by `rvec = θ k`: `p cosθ + (k × p) sinθ + k (k·p)(1 - cosθ)`.
Gauge: fix camera 0 (rotation and translation) and fix scale via the initial pair translation norm.

### A.6 Umeyama Sim(3) alignment
Source points `x_i` (estimated), target `y_i` (reference), `n` points:
`μ_x, μ_y` means; `σ_x² = (1/n) Σ ||x_i - μ_x||²`; `Σ_xy = (1/n) Σ (y_i - μ_y)(x_i - μ_x)^T`; SVD `Σ_xy = U D V^T`; `S = diag(1, 1, det(U) det(V))`; `Q = U S V^T`; `s = trace(D S) / σ_x²`; `b = μ_y - s Q μ_x`. Aligned point: `s Q x + b`.
Applying the transform to a world-to-camera pose `(R, t)`: rotation `R' = R Q^T`; camera center `C' = s Q C + b`. For rotation and direction metrics only `R'` and `C'` matter; scale of `t` is irrelevant. Needs ≥ 3 non-collinear registered cameras.

### A.7 Pose metrics
For image pair `(i, j)` with world-to-camera `(R_i,t_i)`, `(R_j,t_j)`:
- Relative rotation `R_ij = R_j R_i^T`; relative translation `t_ij = t_j - R_ij t_i`.
- Rotation error: `arccos( clip( (trace(R_ij_est^T R_ij_ref) - 1)/2, -1, 1 ) )` in degrees.
- Translation-direction error: angle between `t_ij_est / |t_ij_est|` and `t_ij_ref / |t_ij_ref|` in degrees (scale-free).
- Pair error `e = max(rot_err, trans_err)`. AUC@τ: sort the errors, build the cumulative accuracy curve (recall = (k+1)/n), prepend (0,0), clip at τ degrees with interpolation, integrate by trapezoid, divide by τ. Report τ ∈ {5, 10, 30}. Pairs involving unregistered images count as failures (error = ∞) in the AUC, and also report the registration rate.
- ATE: RMSE of `||C'_est - C_ref||` after Sim(3) alignment.

### A.8 Point cloud metrics
After applying the camera-based Sim(3) to the estimated cloud: subsample each cloud to ≤ 50k points; KD-tree nearest distances (`scipy.spatial.cKDTree`); `accuracy = mean(d_est→ref)`, `completeness = mean(d_ref→est)`, Chamfer = their average; `precision@τ`, `recall@τ`, `F = 2PR/(P+R)`, with `τ = 0.01 × bounding-box diagonal` of the reference (after removing outliers). Caveat to report: different depth sources and sparse clouds differ in density, so compare density-matched subsets and interpret as agreement, not absolute quality.

### A.9 Intrinsics from EXIF
`f_px = f35 / 36 * max(W, H)` where `f35` is `FocalLengthIn35mmFilm` in mm (36 mm is the long side of a 35 mm frame). `cx = W/2`, `cy = H/2`. Apply EXIF orientation first, then compute `W, H`. If images are resized by factor `s`, scale `fx, fy, cx, cy` by `s`.

---

## Appendix B. Default configs

`configs/sfm_default.yaml`
```yaml
seed: 0
image:
  max_side: 1600
features:
  detector: sift          # sift | rootsift | orb
  n_features: 8000
matching:
  ratio: 0.8
  mutual: true
  min_inliers: 30
  ransac_thresh_px: 1.0
  ransac_iters: 2000
  ransac_conf: 0.999
init:
  top_k_pairs: 10
  max_hf_ratio: 0.7
  min_median_angle_deg: 2.0
triangulation:
  max_reproj_px: 2.0
  min_angle_deg: 1.5
registration:
  pnp_reproj_px: 4.0
  pnp_iters: 2000
  min_inliers: 20
ba:
  local_neighbors: 5
  global_every: 5
  loss: huber
  f_scale: 1.0
  max_nfev_local: 30
  max_nfev_global: 100
  max_points_global: 10000
  refine_focal: false
filter:
  final_reproj_px: 3.0
  min_track_length: 2
export:
  save_growth_snapshots: true
```

`configs/eval.yaml`
```yaml
reference: colmap          # colmap | gt  (gt for the public scene)
auc_thresholds_deg: [5, 10, 30]
pointcloud:
  max_points: 50000
  fscore_tau_frac: 0.01    # of reference bbox diagonal
views_ablation: [5, 10, 20, 30]
seeds: [0, 1, 2]           # view-subset sampling seeds for E3
```

---

## Appendix C. Code skeletons

### C.1 Data structures (`reconstruction.py`)
```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Camera:
    image_id: int
    name: str
    K: np.ndarray            # 3x3, pixels
    R: np.ndarray            # 3x3 world-to-camera
    t: np.ndarray            # (3,)
    registered: bool = False

@dataclass
class Reconstruction:
    cameras: dict = field(default_factory=dict)   # image_id -> Camera
    points: dict = field(default_factory=dict)    # track_id -> (3,) XYZ
    tracks: dict = field(default_factory=dict)    # track_id -> {image_id: keypoint_index}
    colors: dict = field(default_factory=dict)    # track_id -> (3,) RGB uint8
    def registered_ids(self): return [i for i, c in self.cameras.items() if c.registered]
```

### C.2 Bundle adjustment core (`bundle_adjust.py`)
```python
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

def rotate(points, rvecs):
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.nan_to_num(rvecs / theta)
    c, s = np.cos(theta), np.sin(theta)
    return c * points + s * np.cross(v, points) + (1 - c) * np.sum(points * v, axis=1)[:, None] * v

def project(points, cam_params, fx, fy, cx, cy):
    p = rotate(points, cam_params[:, :3]) + cam_params[:, 3:6]
    z = p[:, 2]
    u = fx * p[:, 0] / z + cx
    v = fy * p[:, 1] / z + cy
    return np.stack([u, v], axis=1)

def residuals(params, n_cams, n_pts, cam_idx, pt_idx, obs_xy, fx, fy, cx, cy):
    cams = params[: n_cams * 6].reshape(n_cams, 6)
    pts = params[n_cams * 6 : n_cams * 6 + n_pts * 3].reshape(n_pts, 3)
    pred = project(pts[pt_idx], cams[cam_idx], fx, fy, cx, cy)
    return (pred - obs_xy).ravel()

def jac_sparsity(n_cams, n_pts, cam_idx, pt_idx):
    m, n = cam_idx.size * 2, n_cams * 6 + n_pts * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(cam_idx.size)
    for s in range(6):
        A[2 * i, cam_idx * 6 + s] = 1
        A[2 * i + 1, cam_idx * 6 + s] = 1
    for s in range(3):
        A[2 * i, n_cams * 6 + pt_idx * 3 + s] = 1
        A[2 * i + 1, n_cams * 6 + pt_idx * 3 + s] = 1
    return A

# Solve:
# res = least_squares(residuals, x0, jac_sparsity=A, method="trf", loss="huber", f_scale=1.0,
#                     x_scale="jac", tr_solver="lsmr", ftol=1e-4, max_nfev=100, verbose=1,
#                     args=(n_cams, n_pts, cam_idx, pt_idx, obs_xy, fx, fy, cx, cy))
```
Implementation notes:
- Fixed cameras (camera 0, and during local BA all cameras outside the window): do not put them in the parameter vector; keep their params as constants and let the residual function read them from a separate array. A common pattern is to build the parameter vector from the free cameras only and a lookup that maps `cam_idx` to either a free slot or a constant row.
- `fx, fy, cx, cy` are per-observation arrays (so cameras may have different K). If `refine_focal`, add one shared focal parameter at the end of the vector and extend the sparsity matrix with a dense column.
- After BA, write the results back into the `Reconstruction` and recompute reprojection errors; remove observations/points above `final_reproj_px`; run BA once more.
- Verify analytical sanity: residuals at the ground-truth parameters on synthetic noise-free data must be ~0.

### C.3 Feature tracks (`tracks.py`)
Union-find over nodes `(image_id, kp_idx)`. For each verified match `(i, a) ↔ (j, b)` union the two nodes. Group by root. Discard a track if two nodes share an `image_id` (or keep one by the best match score, config flag; default is discard). Optionally discard tracks of length 1. Keep a dict `observation → track_id` for fast 2D-3D lookup in registration.

### C.4 Incremental loop (`incremental.py`) pseudocode
```text
build tracks from verified matches
(i0, i1) = choose_init_pair(match_graph)
recover pose (E, cheirality), triangulate, filter -> initial reconstruction; BA(init)
while True:
    candidates = unregistered images with >= min_inliers 2D-3D correspondences
    if none: break
    j = argmax(# correspondences)
    pose_j = pnp_ransac(...)  # fails -> mark tried, continue
    register j
    triangulate new tracks seen by >= 2 registered cameras
    local_ba(window = j + neighbors)
    if n_registered % global_every == 0: global_ba()
global_ba(); filter outliers; global_ba()
export
```
Retry unregistered images after each success (new points may now make them registrable). Stop when a full pass registers nothing.

### C.5 Checkerboard calibration (`calibrate_camera.py`)
- Inputs: folder of checkerboard images, board inner-corner size (e.g., 9x6), square size in mm.
- `cv2.findChessboardCorners` then `cv2.cornerSubPix(criteria=(EPS+MAX_ITER, 30, 0.001))`; object points `(i*sq, j*sq, 0)`.
- `cv2.calibrateCamera` returns RMS reprojection error, `K`, distortion `dist`. Report RMS (target < 0.7 px), per-image errors, and K.
- Save `calibration.yaml`. To use for SfM, undistort the scene images with `cv2.undistort` using `cv2.getOptimalNewCameraMatrix(K, dist, size, alpha=0)` and use the new K for the undistorted images.
- Use the same lens, same focus lock and same resolution as the scene photos. Calibrate at the same image resolution as used in SfM (scale K if resized).

### C.6 Image preparation (`prepare_data.py`)
- Convert HEIC to JPG (on macOS: `sips -s format jpeg IMG.HEIC --out IMG.jpg`, or `pillow-heif`).
- Apply EXIF orientation: `ImageOps.exif_transpose(img)` before reading `W, H` and before resizing. Phone photos in mixed orientations will break intrinsics otherwise.
- Resize with area interpolation so the long side ≤ `max_side`. Save as high-quality JPG (quality 95) to `images_1600/`.
- Write `intrinsics.yaml` (K per image or one shared K, image sizes, source of K: `exif` | `calibrated`).
- Print a summary: number of images, sizes, focal estimate, any images with missing EXIF.

### C.7 Dense stage (optional)
```python
# rectify pair (1,2): R, T = relative pose cam1 -> cam2 (T keeps reconstruction scale)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
# remap both images with initUndistortRectifyMap + remap
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*8, blockSize=5,
        P1=8*3*5**2, P2=32*3*5**2, uniquenessRatio=10, speckleWindowSize=100,
        speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
pts_rect = cv2.reprojectImageTo3D(disp, Q)          # in rectified cam-1 frame
mask = disp > disp.min() + 1e-3  (also drop far/noisy depth)
X_cam1 = (R1.T @ pts_rect[mask].T).T                # undo rectification rotation
X_world = (R_wc1.T @ (X_cam1 - t1).T).T             # R_wc1, t1: world-to-camera of cam 1
```
Merge pairs, voxel-downsample (Open3D), estimate normals (`estimate_normals`, orient consistently toward cameras), `create_from_point_cloud_poisson(depth=8)`, remove low-density vertices (quantile 0.05). Label this qualitative. Pick pairs with baseline about 5-15% of scene depth and good overlap; pairs with images on the ring are better than near-identical neighbors.

### C.8 Optional PyTorch3D usage (to align with CO2/CO4; skip if install is troublesome on Mac)
On Colab (or Linux) `pip install pytorch3d` per its install guide; use `pytorch3d.loss.chamfer_distance` as a second implementation of the Chamfer metric, or render the point clouds with `PointsRasterizer` for a figure. State in the README whether PyTorch3D is used. It is not required by the project brief.

---

## Appendix D. Synthetic test generator (`tests/synthetic.py`)

```python
def make_scene(n_pts=500, n_cams=10, radius=4.0, arc_deg=80, noise_px=0.5, outlier_frac=0.05,
               W=1280, H=960, f=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_pts, 3))
    K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1.0]])
    cams = []
    for a in np.linspace(-arc_deg/2, arc_deg/2, n_cams):
        th = np.deg2rad(a)
        C = np.array([radius*np.sin(th), -radius*np.cos(th), 0.5])  # on an arc around the origin
        z = -C / np.linalg.norm(C)                                    # look at origin (OpenCV z forward)
        down = np.array([0, 0, -1.0])
        x = np.cross(down, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_cw = np.stack([x, y, z], axis=1)            # camera-to-world (columns)
        R = R_cw.T; t = -R @ C                        # world-to-camera
        cams.append((R, t))
    # project all points, keep those in front of the camera and inside the image
    # add N(0, noise_px^2) to pixel observations; replace outlier_frac of observations by random pixels
    # return K, cams (ground truth), X, observations, and noise-free observations for exact tests
```
The generator must support (a) noise-free exact data, (b) noisy data, (c) putative matches with outliers, (d) a perturbed initialization for BA (add random rotation noise ~2° and translation noise ~5%, point noise ~5%). Check `x × y = z` and `det(R) = +1` in a test of the generator itself.

---

## Appendix E. Local experiment execution

Run Our SfM and COLMAP in the serial experiment workers described in
`docs/EXPERIMENTS.md`. No GPU notebook or multi-view model setup is required.
Keep input hashes, configuration, failed attempts and calibration provenance.

## Appendix F. README.md template

```markdown
# GroupX 3D Computer Vision Project: Our SfM vs. CPU COLMAP

Own incremental Structure-from-Motion (SIFT, RANSAC, triangulation, PnP, sparse bundle adjustment)
benchmarked against CPU COLMAP on self-captured and public scenes.

## Folder structure
(report.pdf, demo.mp4, requirements.txt, code/, sample_data/ ...)

## Install
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(optional) brew install ffmpeg      # only needed to re-render demo video

## Quick demo (about N minutes on a laptop, no GPU)
python code/run_sfm.py --images sample_data/scene_small/images --out outputs/demo --config configs/sfm_default.yaml
python code/eval/run_eval.py --scene sample_data/scene_small --ours outputs/demo --colmap sample_data/scene_small/colmap
Outputs: outputs/demo/points.ply (open with MeshLab/Open3D), poses.json, metrics printed to console.

## Reproduce the experiments
bash code/run_all.sh        # needs data/<scene>/ (see below)
Both camera methods run locally on CPU.

## Large files
Optional Depth Anything V2 Small weights stay outside the archive.
Extra data / precomputed outputs: <Google Drive or HF link, "Anyone with the link can view">

## Third-party code and what we modified
| Component | Version | Use | Our modification |
|---|---|---|---|
| OpenCV | x | SIFT/ORB, BFMatcher, image I/O, solvePnPRansac, SGBM | none (library calls only) |
| pycolmap / COLMAP | x | baseline reconstruction | wrapper scripts only |
| Open3D | x | PLY I/O, visualization, Poisson | none |
| SciPy | x | least_squares for BA | our residual/Jacobian structure |
| Our own code | - | 8-point, RANSAC, E decomposition, DLT triangulation, DLT-PnP, BA, tracks, evaluation | full |

## Team
Names, IDs and a one-line contribution each (full table in report).
```

---

## Appendix G. Report skeleton and drafting plan

### G.1 LaTeX skeleton (`report/main.tex`)
```latex
\documentclass[conference]{IEEEtran}
\usepackage{cite,amsmath,amssymb,graphicx,booktabs,url,hyperref}
\begin{document}
\title{Incremental Structure from Motion: A From-Scratch Pipeline Benchmarked Against CPU COLMAP}
\author{
\IEEEauthorblockN{Member 1}\IEEEauthorblockA{Roll/ID, email} \and
\IEEEauthorblockN{Member 2}\IEEEauthorblockA{Roll/ID, email} \and
\IEEEauthorblockN{Member 3}\IEEEauthorblockA{Roll/ID, email} \and
\IEEEauthorblockN{Member 4}\IEEEauthorblockA{Roll/ID, email}}
\maketitle
\section{Introduction}
\section{Problem Statement}
\section{Literature Review}
\section{Objectives}
\section{Methodology}
\section{Results and Discussion}
\section{Conclusion}
\bibliographystyle{IEEEtran}
\bibliography{refs}
\section*{Author Contributions}
% small table: member / contribution
\end{document}
```
An abstract is optional and not in the required list; include one only if the team wants it and keep it very short. All 9 required elements must be present and in order. Use `\cite{}` for every reference so that numbering is automatic and matches in-text citations.

### G.2 Draft content (facts only; no results until runs exist)
- **Introduction:** 3D reconstruction from photos underpins AR/VR, robotics, mapping, cultural heritage and film. Structure from Motion recovers camera poses and sparse structure from unordered images. Question: how does our transparent incremental pipeline compare with a mature classical system on accuracy, speed and robustness using commodity hardware?
- **Problem statement:** Input: N unordered photos (5-40) of a static scene, approximate or calibrated intrinsics. Output: 6-DoF poses for registered images, sparse colored 3D point cloud (optionally dense cloud/mesh). Evaluation: pose accuracy and agreement with reference, reprojection error, point cloud agreement, runtime, robustness.
- **Objectives (bullets):**
  - Implement a complete incremental SfM pipeline from first principles: SIFT matching, RANSAC F-estimation, essential-matrix pose recovery with cheirality, n-view DLT triangulation, PnP registration, sparse BA with robust loss.
  - Validate every module on synthetic data with known ground truth.
  - Benchmark against CPU COLMAP on self-captured scenes and one public ground-truth scene using ATE, relative-pose AUC, reprojection error, point cloud agreement and runtime.
  - Quantify the effect of bundle adjustment, feature detector, number of input views and intrinsics quality.
  - Characterize failure modes (low texture, planar/small baseline, repetition, drift, learned-model limits).
  - Produce 3D visualizations (orbit renders, incremental growth) and a demo video.
- **Literature review (cover with limitations):** SIFT/ORB (scale invariance, speed vs. robustness); RANSAC and minimal solvers (assumes enough inliers, sensitive to degeneracy); epipolar geometry and SfM (Hartley & Zisserman); incremental SfM systems and COLMAP (robust, but slow, scene-dependent failures, drift without loop closure); bundle adjustment (non-convex, needs good init); monocular depth (relative scale, smoothing and alignment limitations); NeRF and 3DGS (high-quality rendering but need accurate poses and many views). Cite each with a verified reference.
- **Methodology:** pipeline figure, equations from Appendix A, baseline configurations (exact parameters), datasets and capture protocol, metrics (Appendix A.7-A.8), implementation environment (versions, hardware).
- **Results and discussion:** tables and figures listed in G.3; interpret both wins and losses; keep claims scoped to the scenes tested.
- **Conclusion:** summary of the quantitative picture, limitations (no loop closure, small scenes, pseudo-ground-truth on self-captured data, one phone), future work (loop closure and pose graph, learned matchers such as SuperPoint/LightGlue, MVS densification, 3DGS/NeRF from recovered poses).

### G.3 Planned figures and tables (generate by script from `results/`)
- Fig. 1 pipeline diagram (own SfM plus baselines and evaluation)
- Fig. 2 matches and epipolar lines on an image pair
- Fig. 3 camera trajectories (top view) for ours / COLMAP after alignment
- Fig. 4 point cloud renders side by side (S1 and S2)
- Fig. 5 pose AUC (or ATE) vs. number of views for both methods
- Fig. 6 BA ablation: drift/reprojection error without BA vs. local vs. global (loop scene)
- Fig. 7 failure-case panel (glossy objects, low texture, sparse or blurred images, etc.)
- Table I dataset summary (scene, #images, resolution, texture, reference type)
- Table II main results (registration rate, ATE, AUC@5/10/30, reproj error, #points, F-score, runtime)
- Table III ablations (detector, RootSIFT, intrinsics, thresholds)
- Table IV runtime and memory per method and hardware
Keep the total to 4-6 pages: use compact figures, drop low-value ones first.

### G.4 Reference checklist (verify metadata by search before citing)
SIFT (Lowe, 2004); ORB (Rublee et al., 2011); RANSAC (Fischler & Bolles, 1981); Hartley & Zisserman, *Multiple View Geometry*; five-point algorithm (Nistér, 2004); Triggs et al., "Bundle Adjustment: A Modern Synthesis"; COLMAP (Schönberger & Frahm, 2016); DUSt3R (Wang et al., CVPR 2024); Umeyama (1991); Open3D (Zhou et al., 2018); Szeliski book; Middlebury MVS (Seitz et al., 2006) or Strecha et al. (2008) for the chosen public scene; NeRF (Mildenhall et al.) and 3DGS (Kerbl et al.) if discussed; Nerfstudio if used.

---

## Appendix H. Supporting files

`requirements.txt` (pin exact versions after the environment works; only include what `code/` imports)
```text
numpy
scipy
opencv-python
open3d
matplotlib
pyyaml
tqdm
pandas
pillow
imageio
imageio-ffmpeg
exifread
pycolmap
# torch            # optional monocular depth
```

`code/run_all.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SCENES=(s1 s2 gt_scene)
for s in "${SCENES[@]}"; do
  python code/prepare_data.py --scene "$s"
  python code/run_sfm.py --images "data/$s/images_1600" --out "results/$s/ours" --config configs/sfm_default.yaml
  python code/baselines/run_colmap.py --images "data/$s/images_1600" --out "results/$s/colmap"
  python code/eval/run_eval.py --scene "$s" --config configs/eval.yaml
done
python code/eval/make_tables.py
```

`.gitignore`
```text
.venv/
__pycache__/
*.pyc
.DS_Store
.ipynb_checkpoints/
data/
results/
outputs/
*.npz
!sample_data/**/*.npz
*.ply
!sample_data/**/*.ply
```

---

## Appendix I. Team workflow and interface contracts

- One shared Git repo (private). `main` always runs. Each member works on a branch named `feat/<module>` and merges via pull request with at least the unit tests passing. Commit messages state the module and what changed.
- **Interface contracts** (so four people can work in parallel from Day 1):
  - Common pose format (Section 5) is the only way methods talk to evaluation.
  - `features.py`: `extract_features(img, detector, n) -> (kps[N,2], desc[N,D])`.
  - `matching.py`: `match_images(desc1, desc2, ratio, mutual) -> idx_pairs[M,2]`; `verify_pair(kps1, kps2, idx_pairs, cfg) -> (F, inlier_idx)`.
  - `geometry.py`: `fundamental_8pt`, `ransac_fundamental`, `essential_from_F`, `decompose_essential`, `triangulate_dlt`, `pnp_dlt`, `reprojection_error`.
  - `incremental.py`: `run_sfm(images_dir, config) -> Reconstruction`.
  - `io_utils.py`: `save_poses_json(recon, path)`, `load_poses_json(path)`, `save_ply(points, colors, path)`.
  - `eval/run_eval.py` consumes `poses.json` and PLY files only.
- Daily 15-minute sync: what passed, what is blocked, who needs what. Keep `NOTES.md` current.
- Everyone must be listed in the report's Author Contributions with true work. All members must be tagged on the portal.

---

## Appendix J. Troubleshooting FAQ

| Symptom | Likely cause and fix |
|---|---|
| Intrinsics look wrong; reconstructions are skewed | EXIF orientation not applied, or images of mixed orientation. Apply `exif_transpose` before reading W, H; check that all images have consistent sizes |
| HEIC files will not open | Convert with `sips -s format jpeg` or `pillow-heif` |
| Few SIFT features on dark or flat images | Apply CLAHE to the grayscale image before SIFT; lower the contrast threshold; use more light |
| Too many wrong matches (repetitive scenes) | Lower ratio to 0.75, keep mutual check, raise min_inliers; report as a failure case |
| Essential decomposition picks wrong pose | Verify cheirality counts; check that `det(R) = +1`; confirm K is correct; confirm the x2 = R x1 + t convention |
| Initialization collapses (everything near the camera) | Pair has tiny baseline or is planar; choose another pair using the H/F ratio and triangulation angle |
| PnP fails for many images | Too few 3D points; triangulate more tracks first; relax reprojection error to 6 px; retry later in the loop |
| BA diverges or is very slow | Check residuals at ground truth on synthetic data; use Huber loss; limit points; local BA first; make sure fixed-camera handling and `x_scale="jac"` are used; reduce `max_nfev` |
| Reconstruction drifts around a loop | Expected without loop closure; this is the E2 ablation; mention pose-graph/loop closure as future work |
| Reconstruction scale differs between methods | Expected; always Sim(3)-align before metrics |
| COLMAP produces several models | Pick the largest; report the number of models; this is a robustness data point |
| pycolmap API errors | Check the installed version's docs; fall back to the COLMAP CLI via `subprocess` |
| Dense reconstruction uses too much memory | Lower depth resolution / grid, bound views and process one job at a time |
| Colab disconnects | Save outputs to Drive after each run; make cells idempotent |
| Open3D window or offscreen rendering fails on macOS | Run from the main thread, use `Visualizer` with a window and `capture_screen_image`; or use the matplotlib fallback |
| Pose conversion mismatch | Add round-trip tests; check COLMAP `cam_from_world` vs. world-from-camera; check Middlebury/Strecha rotation convention |
| zip contains `__MACOSX` or `.DS_Store` | Use the exclusion flags from Section 14 |
| Results not reproducible | Seed everything, save configs with results, and avoid nondeterministic thread-dependent steps in tests |

---

## Appendix K. Stretch: Nerfstudio / 3DGS export (only if all "Must" work is done)

- Goal: a short fly-through video from our own recovered poses; optional comparison NeRF (`nerfacto`) vs. 3DGS (`splatfacto`) on S1.
- Export `transforms.json` from our reconstruction:
  - Per frame: `file_path`, `transform_matrix` = camera-to-world in **OpenGL/Nerfstudio convention**. Compute `c2w_cv` from `(R, t)` (`c2w = [[R^T, -R^T t],[0,1]]`), then `c2w_gl = c2w_cv @ diag(1, -1, -1, 1)`.
  - Top-level: `fl_x, fl_y, cx, cy, w, h` (and distortion coefficients if any), `camera_model: "OPENCV"` or `"PINHOLE"`.
  - Initial points (`ply_file_path`) can be our sparse cloud.
- Train on Colab T4 for a short schedule (about 20-40 min), keep images ≤ 1600 px wide, save checkpoints to Drive. Record PSNR/SSIM/LPIPS on held-out views if time permits.
- Cite Nerfstudio (and the 3DGS paper) in the report; list in README what was modified (only data export and run scripts). Hard time cap of half a day.

---

## Appendix L. Glossary and open items for humans

**Glossary:** SfM structure from motion; BA bundle adjustment; PnP perspective-n-point; DLT direct linear transform; RANSAC random sample consensus; F/E fundamental/essential matrix; Sim(3) similarity transform (rotation, translation, scale); ATE absolute trajectory error; AUC area under the cumulative error curve; RPE relative pose error; MVS multi-view stereo; SGBM semi-global block matching; TSDF truncated signed distance function; 3DGS 3D Gaussian Splatting; NeRF neural radiance field; pseudo-GT reference produced by another algorithm, not by measurement.

**Open items (agent: ask the humans for these early):**
1. Group number X (zip and folder name).
2. Member names, roll numbers/IDs, emails; who owns which module (suggestion in Section 15).
3. Phone model and whether the images are HEIC or JPG; Mac RAM.
4. Which scenes will be photographed (S1, S2, S3) and by when; whether a printed checkerboard is available.
5. Whether anyone has a local LaTeX install or will use Overleaf.
6. Whether the course portal imposes anything not in the brief (file naming, member tagging process).
7. Any instructor approval needed for the project topic (not stated in the brief; ask if unsure).
