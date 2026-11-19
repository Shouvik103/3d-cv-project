"""Convert COLMAP reconstructions into common poses and PLY."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime import write_json
from sfm.io_utils import save_ply


def export_colmap(model, out, all_names=None):
    names = sorted(image.name for image in model.images.values())
    poses = dict(image_names=sorted(all_names or names), registered=names, K={}, R={}, t={}, image_size={},distortion={})
    for image in model.images.values():
        camera = model.cameras[image.camera_id]
        transform = image.cam_from_world()
        poses['R'][image.name] = transform.rotation.matrix().tolist()
        poses['t'][image.name] = transform.translation.tolist()
        poses['K'][image.name] = camera.calibration_matrix().tolist()
        poses['image_size'][image.name] = [camera.width, camera.height]
        if camera.model_name=='SIMPLE_RADIAL':poses['distortion'][image.name]=[float(camera.params[3]),0,0,0,0]
        elif camera.model_name=='RADIAL':poses['distortion'][image.name]=[float(camera.params[3]),float(camera.params[4]),0,0,0]
    write_json(out/'poses.json', poses)
    points = list(model.points3D.values())
    save_ply(np.array([p.xyz for p in points]), np.array([p.color for p in points]), out/'points.ply')
    observations = {}
    for point in points:
        track = {}
        for element in point.track.elements:
            image = model.images[element.image_id]
            track[image.name] = image.points2D[element.point2D_idx].xy.tolist()
        observations[str(point.point3D_id) if hasattr(point, 'point3D_id') else str(len(observations))] = track
    write_json(out/'observations.json', observations)
    return poses




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--colmap',type=Path,required=True)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    import pycolmap
    export_colmap(pycolmap.Reconstruction(a.colmap),a.out)


if __name__ == '__main__': main()
