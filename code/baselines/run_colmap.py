"""CPU sparse COLMAP baseline on exactly the supplied images and calibration."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from runtime import manifest, write_json
from baselines.convert_to_common import export_colmap


def run(images, out, min_matches=15, progress=None,object_quality=False):
    import pycolmap
    if (out/'database.db').exists():
        raise ValueError('Output already contains a COLMAP database; choose a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    calibration = images.parent/'intrinsics.yaml'
    if not calibration.exists(): raise ValueError('Run prepare_data.py first to provide intrinsics.yaml')
    records = yaml.safe_load(calibration.read_text())['images']
    Ks = [np.array(records[n]['K']) for n in names]
    shared=all(np.allclose(Ks[0], K) for K in Ks)
    if not shared and not object_quality:
        raise ValueError('This baseline uses a shared PINHOLE camera; supply images with identical K')
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'PINHOLE'
    K = Ks[0]; reader.camera_params = ','.join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    extraction = pycolmap.FeatureExtractionOptions(); extraction.num_threads = 4
    estimated=not all(records[n].get('source','').startswith(('user_','supplied','checkerboard')) for n in names)
    if object_quality:
        extraction.sift.max_num_features=8000
        masks=images.parent/'masks'
        if masks.exists():reader.mask_path=str(masks)
    matching = pycolmap.FeatureMatchingOptions(); matching.num_threads = 4
    if object_quality:matching.guided_matching=True
    def report(phase,done,total,message):
        if progress:progress(phase,done,total,message)
    report('features',0,1,'COLMAP: extracting SIFT features on CPU')
    if shared:
        groups=[names]
    else:
        by_camera={}
        for name,K in zip(names,Ks):
            identity=(*records[name]['image_size'],*K.ravel())
            by_camera.setdefault(identity,[]).append(name)
        groups=list(by_camera.values())
    for group in groups:
        K=np.asarray(records[group[0]]['K'])
        reader.camera_params=','.join(map(str,[K[0,0],K[1,1],K[0,2],K[1,2]]))
        # Let COLMAP create a consistent camera/rig/frame group at extraction;
        # changing image.camera_id afterwards invalidates COLMAP 4.x frames.
        pycolmap.extract_features(database_path=out/'database.db', image_path=images,image_names=group,
                                 camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    report('features',1,1,'COLMAP: features extracted')
    report('matching',0,1,'COLMAP: exhaustive matching on CPU')
    pycolmap.match_exhaustive(database_path=out/'database.db', matching_options=matching, device=pycolmap.Device.cpu)
    report('matching',1,1,'COLMAP: matching complete')
    report('registration',0,len(names),'COLMAP: incremental mapping and bundle adjustment')
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = False; options.ba_refine_principal_point = False; options.ba_refine_extra_params = False
    if object_quality and estimated:options.ba_refine_focal_length=True
    if object_quality:
        options.num_threads=4;options.mapper.num_threads=4
        options.random_seed=0;options.mapper.random_seed=0;options.triangulation.random_seed=0
        options.mapper.init_min_tri_angle=8.
        options.mapper.abs_pose_refine_focal_length=estimated
        options.mapper.abs_pose_refine_extra_params=False
    options.min_num_matches = min_matches
    models = pycolmap.incremental_mapping(database_path=out/'database.db', image_path=images, output_path=out/'sparse', options=options)
    if not models: raise RuntimeError('COLMAP did not reconstruct a model')
    model = max(models.values(), key=lambda m: m.num_reg_images())
    poses = export_colmap(model, out, names)
    report('registration',len(poses['registered']),len(names),'COLMAP: largest reconstructed model exported')
    write_json(out/'summary.json', dict(manifest(dict(camera_model='PINHOLE', fixed_intrinsics=not options.ba_refine_focal_length, device='cpu',object_quality=object_quality,max_sift_features=extraction.sift.max_num_features,guided_matching=matching.guided_matching,num_threads=options.num_threads)),
                                       registered=len(poses['registered']), total_images=len(names), points=model.num_points3D(),
                                       runtime_seconds=time.perf_counter()-started, models=len(models),
                                       mean_reprojection_px=float(model.compute_mean_reprojection_error())))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--images', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); run(a.images, a.out)
