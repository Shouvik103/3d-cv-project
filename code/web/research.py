"""Read-only, dependency-light access to versioned experiment evidence and runs."""
from pathlib import Path
import json
from urllib.parse import quote

STAGES = {
    'E1': ('Method comparison', 'Our SfM, COLMAP and VGGT on identical car, fountain and Temple inputs.'),
    'E2': ('Bundle adjustment', 'No BA, local BA, and local + global BA on car and calibrated Temple.'),
    'E3': ('Number of views', 'Car: 5/10/20/30 views. Temple: 5/10/16. Missing cameras count against pose AUC.'),
    'E4': ('Feature detectors', 'SIFT, RootSIFT and ORB, with other parameters held fixed.'),
    'E5': ('Camera intrinsics', 'Approximate K, supplied calibration, and optional shared-focal BA on Temple.'),
    'E6': ('Matching sensitivity', 'One-factor changes to the ratio and RANSAC thresholds on the car.'),
    'E7': ('Failure cases', 'Glossy car, sparse views, controlled Gaussian blur and low-texture Dino.'),
    'M4': ('Ground-truth mesh', 'Original fountain-P11 scan, camera-only Sim(3), GT-box crop and 200k surface samples.'),
    'M5': ('Mesh source comparison', 'Our / COLMAP cameras with the same mono depth backend; VGGT with its own depth.'),
}
METHODS = [
    dict(id='ours', name='Our SfM', label='Geometry, step by step', device='CPU',
         description='Incremental structure from motion built from feature correspondences, triangulation and bundle adjustment.',
         steps=['Prepare images + intrinsics', 'SIFT / RootSIFT / ORB features', 'Mutual matching + ratio test', 'RANSAC epipolar geometry', 'Initial pair + triangulation', 'PnP camera registration', 'Local + global bundle adjustment', 'Filter and export sparse geometry'],
         outputs='Camera poses, sparse colored points, observed tracks, reprojection residuals and growth history.',
         mesh='Uploads offer SGBM stereo or pinned Depth Anything V2 Small + sparse depth alignment, followed by TSDF. M4/M5 used the monocular option.',
         limits='Depends on texture, sharpness and overlap. Partial registration, bad initialization and globally wrong geometry can remain despite small pixel residuals. No loop closure or multicomponent merging.',
         calibration='Standard runs keep supplied/estimated K fixed. Object quality refines shared focal in global BA only for estimated K; it did not recover the correct Temple calibration in E5.',
         source='https://github.com/Shouvik103/3d-cv-project',license='Project implementation; third-party library terms apply.'),
    dict(id='colmap', name='COLMAP', label='Classical reference baseline', device='CPU · pycolmap 4.2.1',
         description='A separate classical SfM baseline using the same prepared image files and PINHOLE calibration. Object quality handles portrait/landscape camera groups and refines estimated focal lengths.',
         steps=['Read the shared prepared images', 'Extract SIFT features', 'Match and verify pairs', 'Incremental mapping + BA', 'Choose the largest model', 'Convert to common pose / point exports'],
         outputs='Registered cameras, sparse points, observed tracks and reprojection statistics. Disconnected models are reported; only the largest is exported.',
         mesh='M5 feeds COLMAP cameras and landmarks to the same monocular depth / TSDF policy as Our SfM. This is not COLMAP dense MVS.',
         limits='Can fail or split into components when viewpoint overlap is weak. Car/fountain full-data COLMAP cameras are pseudo-references, not independent ground truth.',
         calibration='E1 uses the same fixed PINHOLE K as Our SfM. Earlier full-data car/fountain pseudo-references had refined intrinsics.',
         source='https://github.com/colmap/colmap',license='COLMAP BSD license; linked dependency terms apply.'),
    dict(id='vggt', name='VGGT', label='Joint learned reconstruction', device='Apple MPS · VGGT-1B',
         description='A pretrained transformer jointly predicts cameras, intrinsics, depths and world points across the input views.',
         steps=['Official 518px crop preprocessing', 'Joint multi-view aggregation', 'Predict camera poses + intrinsics', 'Predict depth + world points', 'Save confidence and raw arrays', 'Convert to common exports'],
         outputs='Predicted cameras, intrinsics, dense depth, world points and confidence. Camera count means predictions, not feature-verified registration.',
         mesh='M5 fuses VGGT’s own depth, confidence and cameras. It does not mix VGGT depth with unrelated SfM poses.',
         limits='Predictions can be inaccurate or incomplete. Different preprocessing and depth models confound direct comparisons. Object quality pads to preserve the full object. Inputs above 16 views use a global overview and overlapping batches with checked camera alignment, which can introduce seams. No observed feature tracks means no comparable pixel reprojection statistic.',
         calibration='Predicts its own K after official crop preprocessing. MPS uses FP16 aggregation / FP32 heads, one-frame dense chunks, and no unused tracking head; no CUDA timings are claimed.',
         source='https://github.com/facebookresearch/vggt',license='Original VGGT-1B checkpoint: CC-BY-NC-4.0. Weights are not distributed.'),
]
GLOSSARY = [
    ('Registration', 'Cameras reconstructed / input views. VGGT returns predictions; these are not feature-verified registrations.'),
    ('ATE (%) ↓', 'Camera-center RMSE after Sim(3), divided by reference trajectory extent. Uses shared cameras only; undefined alignment stays missing.'),
    ('Pose AUC@5 ↑', 'Relative-pose accuracy integrated up to 5°. Missing camera pairs are penalized. Range 0–1.'),
    ('Reprojection (px) ↓', 'Median pixel residual on observed feature tracks at processed image resolution; not a full 3D accuracy measure.'),
    ('Accuracy / completeness ↓', 'Mean estimate→GT / GT→estimate surface distance, as a percentage of the GT mesh diagonal.'),
    ('F-score ↑', 'Precision/recall balance at 0.5%, 1% or 2% of the GT diagonal. Range 0–1; not an object-completeness percentage.'),
    ('Coverage', 'Fraction of held-out image pixels hit by rendered geometry. Read beside masked PSNR/SSIM; small patches can score well.'),
    ('Timing & memory', 'E1 time is end-to-end method runtime; mesh time is fusion pipeline only, excluding upstream SfM/VGGT inference and GT scoring. Peak process RSS is not GPU memory. Caches differ.'),
    ('Exported points & cloud agreement', 'VGGT common exports retain up to 50k predicted points, not the full dense-array count. Cloud distance/F-score compare against COLMAP sparse points in arbitrary aligned units; they are not GT surface accuracy.'),
]


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return default


def evidence_root(project):
    candidates=sorted((Path(project)/'validation').glob('experiments_*'),reverse=True)
    return next((p for p in candidates if (p/'evidence_manifest.json').is_file() and (p/'experiment_status.json').is_file()),None)


def asset_path(project, relative):
    base=evidence_root(project)
    if not base: raise ValueError('Saved experiment evidence is not available')
    path=(base/relative).resolve()
    allowed={item['path'] for item in read(base/'evidence_manifest.json',{}).get('files',[])}
    if not path.is_relative_to(base.resolve()) or relative not in allowed or not path.is_file():
        raise ValueError('Unknown experiment asset')
    return path


def catalog(project):
    base=evidence_root(project)
    output=dict(available=bool(base),methods=METHODS,glossary=GLOSSARY,experiments=[],scenes=[
        dict(id='car_s1',name='Car',reference='COLMAP agreement · not ground truth'),
        dict(id='fountain_s2',name='Fountain',reference='E1: COLMAP agreement · M4/M5: real GT mesh'),
        dict(id='temple_gt',name='Temple',reference='Supplied ground-truth cameras'),
    ])
    if not base: return output
    ledger=read(base/'experiment_status.json',{}).get('experiments',{})
    records=read(base/'evidence_manifest.json',{}).get('files',[])
    assets=[dict(path=r['path'],url='/api/research/files/'+quote(r['path'],safe='/'),bytes=r['bytes']) for r in records]
    for key,(title,description) in STAGES.items():
        rows=read(base/key/'results.json',[])
        # Missing values remain null. Failures and partial cameras are not dropped.
        rows=[dict(row,method=row.get('method','ours'),total=row.get('total',row.get('views'))) for row in rows]
        output['experiments'].append(dict(id=key,title=title,description=description,status=ledger.get(key,{}).get('status','unavailable'),
            rows=rows,assets=[a for a in assets if a['path'].startswith(key+'/')]))
    output.update(collection=base.name,date='18 November 2026' if base.name=='experiments_20261118' else base.name,
                  verification=read(base/'verification.json',{}),
                  vggt=read(base/'E1/temple_gt/vggt/raw.json',{}),
                  notes=[
                      'The benchmark tabs show recorded development experiments. Your comparisons shows fresh same-upload runs; method selection is available in Workspace.',
                      'Car curation previously used COLMAP poses; the relaxed SfM configuration was tuned on Temple. These are not unseen generalization results.',
                      'E5 uses calibrated Temple as approved because no real car checkerboard calibration was available. Shared-focal BA did not recover the true calibration.',
                      'Mesh scores use the full fountain GT scan without visibility masking. Missing or unseen scan regions affect completeness; this is not the official EPFL scorer.',
                      'M5 camera, sparse-landmark, ROI and depth-alignment effects are coupled. VGGT also uses a different depth model/resolution. Fusion-held-out images participated in pose/network inference.',
                  ])
    return output


def run_analysis(folder,metadata,state):
    """Selected-run details only; never infer GT accuracy for an uploaded object."""
    folder=Path(folder);sfm=read(folder/'reconstruction/summary.json',{})
    mesh=read(folder/'mesh/summary.json',{})
    from web.settings import METHOD_NAMES
    method=metadata.get('method','ours')
    return dict(name=metadata['name'],method=METHOD_NAMES.get(method,method),mesh_backend=mesh.get('backend',('vggt' if method=='vggt' else metadata.get('depth_backend','sgbm')) if metadata.get('make_mesh') else None),
        status=state.get('status'),elapsed_seconds=state.get('elapsed_seconds'),cloud_scope=state.get('metrics',{}).get('cloud_scope'),geometry_reused=state.get('metrics',{}).get('geometry_reused',False),
        settings={key:metadata.get(key) for key in ['method','max_side','focal_px','camera_matrix','matching_ratio','depth_backend','make_mesh','mesh_quality','quality_profile','reuse_geometry']},prior_attempts=metadata.get('prior_attempts',0),
        reconstruction={key:sfm.get(key) for key in ['total_images','registered','points','median_reprojection_px','mean_reprojection_px','mean_track_length','runtime_seconds','init_pair','focal_refinement','registration_scope','cloud_scope','memory_scope','vggt']},
        history=sfm.get('history',[]),configuration=sfm.get('config',{}),versions=sfm.get('versions',{}),
        mesh={key:mesh.get(key) for key in ['backend','registered_views','frames_fused','full','light','timing_seconds','heldout','peak_rss_mb','depth_samples','tsdf_cloud_points']},
        warnings=state.get('warnings',[]),reference='No ground-truth cameras or surface were supplied for this upload. ATE and GT F-scores are unavailable.')
