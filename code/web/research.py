"""Read-only, dependency-light access to versioned experiment evidence and runs."""
from pathlib import Path
import json
from urllib.parse import quote

STAGES = {
    'E1': ('Method comparison', 'Our SfM and COLMAP on identical car, fountain and Temple inputs.'),
    'E2': ('Bundle adjustment', 'No BA, local BA, and local + global BA on car and calibrated Temple.'),
    'E3': ('Number of views', 'Car: 5/10/20/30 views. Temple: 5/10/16. Missing cameras count against pose AUC.'),
    'E4': ('Feature detectors', 'SIFT, RootSIFT and ORB, with other parameters held fixed.'),
    'E5': ('Camera intrinsics', 'Approximate K, supplied calibration, and optional shared-focal BA on Temple.'),
    'E6': ('Matching sensitivity', 'One-factor changes to the ratio and RANSAC thresholds on the car.'),
    'E7': ('Failure cases', 'Glossy car, sparse views, controlled Gaussian blur and low-texture Dino.'),
    'M4': ('Ground-truth mesh', 'Original fountain-P11 scan, camera-only Sim(3), GT-box crop and 200k surface samples.'),
    'M5': ('Mesh source comparison', 'Our SfM and COLMAP cameras with the same mono depth backend.'),
}
FINDINGS = {
    'E1': 'On Temple, Our SfM registered 16/16 cameras (0.270% ATE, 0.864 AUC@5). COLMAP registered 13/16 (0.094% ATE, 0.606 AUC@5). COLMAP ran faster on all three inputs. ATE measures only shared cameras.',
    'E2': 'On Temple, local and global BA reduced ATE from 0.377% to 0.270%. On the car, lower reprojection error did not resolve incorrect global geometry or missing cameras.',
    'E3': 'More views did not always improve registration. Our SfM registered 10/20 car views but only 2/30 on the independently sampled 30-view set. Global ATE is unavailable for the two-camera result.',
    'E4': 'ORB failed to initialize on the car but registered all 11 fountain cameras. The detector choice depended on the scene.',
    'E5': 'Temple ATE was 0.270% with supplied calibration, 6.970% with approximate intrinsics, and 8.297% after shared-focal BA. The real run did not recover true calibration. The separate synthetic recovery test passed.',
    'E6': 'A ratio of 0.7 reduced car ATE against the COLMAP reference to 0.935%, while retaining only 10/20 cameras. The frozen E1 baseline still uses its original threshold.',
    'E7': 'Both methods failed on the five-view car and deliberately blurred Temple. On Dino, Our SfM registered 4/16 cameras and COLMAP registered 6/16.',
    'M4': 'The Our SfM mesh fused eight views and achieved 3.205% surface accuracy error, 10.087% completeness error, and 0.059 F-score at 1% of the GT diagonal. The score includes unseen scan regions.',
    'M5': 'With the shared monocular depth and fusion policy, COLMAP had lower accuracy and completeness errors and higher F@1% (0.098 vs. 0.059). Both surfaces were incomplete. Camera poses, sparse landmarks, and ROI effects are coupled.',
}
METHODS = [
    dict(id='ours', name='Our SfM', label='Geometry, step by step', device='CPU',
         description='Incremental structure from motion built from feature correspondences, triangulation and bundle adjustment.',
         steps=['Prepare images + intrinsics', 'SIFT / RootSIFT / ORB features', 'Mutual matching + ratio test', 'RANSAC epipolar geometry', 'Initial pair + triangulation', 'PnP camera registration', 'Local + global bundle adjustment', 'Filter and export sparse geometry'],
         outputs='Camera poses, sparse colored points, observed tracks, reprojection residuals and growth history.',
         mesh='Uploads offer SGBM stereo or pinned Depth Anything V2 Small + sparse depth alignment, followed by TSDF. M4/M5 used the monocular option.',
         limits='Depends on texture, sharpness and overlap. Partial registration, bad initialization and globally wrong geometry can remain despite small pixel residuals. No loop closure or multicomponent merging.',
         calibration='Standard runs keep K fixed. Object quality refines shared focal only for estimated K; Maximum quality can lock an explicit intrinsic prior. Shared-focal BA did not recover the correct Temple calibration in E5.',
         source='https://github.com/Shouvik103/3d-cv-project',license='Project implementation; third-party library terms apply.'),
    dict(id='colmap', name='COLMAP', label='Classical reference baseline', device='CPU · pycolmap 4.2.1',
         description='A separate classical SfM baseline using the same uploaded photos. Benchmark runs use PINHOLE calibration; Maximum quality supports shared SIMPLE_RADIAL cameras, lens priors and affine/domain-size pooled SIFT.',
         steps=['Read the shared prepared images', 'Extract SIFT features', 'Match and verify pairs', 'Incremental mapping + BA', 'Choose the largest model', 'Convert to common pose / point exports'],
         outputs='Registered cameras, sparse points, observed tracks and reprojection statistics. Disconnected models are reported; only the largest is exported.',
         mesh='M5 feeds COLMAP cameras and landmarks to the same monocular depth / TSDF policy as Our SfM. This is not COLMAP dense MVS.',
         limits='Can fail or split into components when viewpoint overlap is weak. Car/fountain full-data COLMAP cameras are pseudo-references, not independent ground truth.',
         calibration='E1 uses the same fixed PINHOLE K as Our SfM. Earlier full-data car/fountain pseudo-references had refined intrinsics.',
         source='https://github.com/colmap/colmap',license='COLMAP BSD license; linked dependency terms apply.'),

]
GLOSSARY = [
    ('Registration', 'Cameras reconstructed / input views.'),
    ('ATE (%) ↓', 'Camera-center RMSE after Sim(3), divided by reference trajectory extent. Uses shared cameras only; undefined alignment stays missing.'),
    ('Pose AUC@5 ↑', 'Relative-pose accuracy integrated up to 5°. Missing camera pairs are penalized. Range 0–1.'),
    ('Pose AUC@10 / AUC@30 ↑', 'The same relative-pose measure at more permissive cutoffs. Rotation and translation columns report median pairwise angular errors in degrees.'),
    ('Reprojection (px) ↓', 'Median pixel residual on observed feature tracks at processed image resolution; not a full 3D accuracy measure.'),
    ('Accuracy / completeness ↓', 'Mean estimate→GT / GT→estimate surface distance, as a percentage of the GT mesh diagonal.'),
    ('F-score ↑', 'Precision/recall balance at 0.5%, 1% or 2% of the GT diagonal. Range 0–1; not an object-completeness percentage.'),
    ('Coverage', 'Fraction of held-out image pixels hit by rendered geometry. Read beside masked PSNR/SSIM; small patches can score well.'),
    ('Timing & memory', 'E1 time is end-to-end method runtime; mesh time is fusion pipeline only, excluding upstream camera estimation and GT scoring. Peak process RSS measures host memory. Caches differ.'),
    ('Exported points & cloud agreement', 'Cloud distance/F-score compare against COLMAP sparse points in arbitrary aligned units; they are not GT surface accuracy.'),
]


def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except (OSError, ValueError): return default


def evidence_root(project):
    candidates=sorted((Path(project)/'validation').glob('experiments_*'),reverse=True)
    return next((p for p in candidates if (p/'evidence_manifest.json').is_file() and (p/'experiment_status.json').is_file() and read(p/'evidence_manifest.json',{}).get('methods')==['ours','colmap']),None)


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
        rows=[dict(row,method=row.get('method','ours'),total=row.get('total',row.get('views'))) for row in rows if row.get('method','ours') in ('ours','colmap')]
        finding = FINDINGS[key] if base.name == 'experiments_20261119_classical' else ''
        output['experiments'].append(dict(id=key,title=title,description=description,finding=finding,status=ledger.get(key,{}).get('status','unavailable'),
            rows=rows,assets=[a for a in assets if a['path'].startswith(key+'/')]))
    output.update(collection=base.name,date='18 November 2026 · classical projection 19 November 2026',
                  verification=read(base/'verification.json',{}),
                  artifacts=[a for a in assets if '/' not in a['path'] or a['path'].startswith('datasets/')],
                  notes=[
                      'The active two-method collection retains the 18 November development measurements. The 19 November projection removed retired method rows and regenerated plots without rerunning reconstruction.',
                      'Car curation previously used COLMAP poses; the relaxed SfM configuration was tuned on Temple. These are not unseen generalization results.',
                      'E5 uses calibrated Temple as approved because no real car checkerboard calibration was available. Shared-focal BA did not recover the true calibration.',
                      'Mesh scores use the full fountain GT scan without visibility masking. Missing or unseen scan regions affect completeness; this is not the official EPFL scorer.',
                      'M5 camera, sparse-landmark, ROI and depth-alignment effects are coupled. Fusion-held-out images participated in camera estimation.',
                  ])
    return output


def run_analysis(folder,metadata,state):
    """Selected-run details only; never infer GT accuracy for an uploaded object."""
    folder=Path(folder);sfm=read(folder/'reconstruction/summary.json',{})
    mesh=read(folder/'mesh/summary.json',{})
    from web.settings import METHOD_NAMES
    method=metadata.get('method','ours')
    return dict(name=metadata['name'],method=METHOD_NAMES.get(method,method),mesh_backend=mesh.get('backend',metadata.get('depth_backend','sgbm') if metadata.get('make_mesh') else None),
        status=state.get('status'),elapsed_seconds=state.get('elapsed_seconds'),cloud_scope=state.get('metrics',{}).get('cloud_scope'),geometry_reused=state.get('metrics',{}).get('geometry_reused',False),
        settings={key:metadata.get(key) for key in ['method','max_side','focal_px','camera_matrix','matching_ratio','depth_backend','make_mesh','mesh_quality','quality_profile','reuse_geometry','camera_prior']},prior_attempts=metadata.get('prior_attempts',0),
        reconstruction={key:sfm.get(key) for key in ['total_images','registered','points','median_reprojection_px','mean_reprojection_px','mean_track_length','runtime_seconds','init_pair','focal_refinement','registration_scope','cloud_scope','memory_scope']},
        diagnostics=read(folder/'registration_diagnostics.json',{}),camera_prior=read(folder/'camera_prior.json',{}),history=sfm.get('history',[]),configuration=sfm.get('config',{}),versions=sfm.get('versions',{}),
        mesh={key:mesh.get(key) for key in ['backend','registered_views','frames_fused','full','light','timing_seconds','heldout','peak_rss_mb','depth_samples','tsdf_cloud_points']},
        warnings=state.get('warnings',[]),reference='No ground-truth cameras or surface were supplied for this upload. ATE and GT F-scores are unavailable.')
