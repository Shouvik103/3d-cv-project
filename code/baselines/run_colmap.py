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


def run(images, out, min_matches=15, progress=None):
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
    if not all(np.allclose(Ks[0], K) for K in Ks):
        raise ValueError('This baseline uses a shared PINHOLE camera; supply images with identical K')
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = 'PINHOLE'
    K = Ks[0]; reader.camera_params = ','.join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    extraction = pycolmap.FeatureExtractionOptions(); extraction.num_threads = 4
    matching = pycolmap.FeatureMatchingOptions(); matching.num_threads = 4
    def report(phase,done,total,message):
        if progress:progress(phase,done,total,message)
    report('features',0,1,'COLMAP: extracting SIFT features on CPU')
    pycolmap.extract_features(database_path=out/'database.db', image_path=images,
                             camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    report('features',1,1,'COLMAP: features extracted')
    report('matching',0,1,'COLMAP: exhaustive matching on CPU')
    pycolmap.match_exhaustive(database_path=out/'database.db', matching_options=matching, device=pycolmap.Device.cpu)
    report('matching',1,1,'COLMAP: matching complete')
    report('registration',0,len(names),'COLMAP: incremental mapping and bundle adjustment')
    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = False; options.ba_refine_principal_point = False; options.ba_refine_extra_params = False
    options.min_num_matches = min_matches
    models = pycolmap.incremental_mapping(database_path=out/'database.db', image_path=images, output_path=out/'sparse', options=options)
    if not models: raise RuntimeError('COLMAP did not reconstruct a model')
    model = max(models.values(), key=lambda m: m.num_reg_images())
    poses = export_colmap(model, out, names)
    report('registration',len(poses['registered']),len(names),'COLMAP: largest reconstructed model exported')
    write_json(out/'summary.json', dict(manifest(dict(camera_model='PINHOLE', fixed_intrinsics=True, device='cpu')),
                                       registered=len(poses['registered']), total_images=len(names), points=model.num_points3D(),
                                       runtime_seconds=time.perf_counter()-started, models=len(models),
                                       mean_reprojection_px=float(model.compute_mean_reprojection_error())))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--images', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); run(a.images, a.out)
