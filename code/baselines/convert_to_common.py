"""Convert COLMAP reconstructions or VGGT NPZ output into common poses and PLY."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime import write_json
from sfm.io_utils import save_ply


def export_colmap(model, out, all_names=None):
    names = sorted(image.name for image in model.images.values())
    poses = dict(image_names=sorted(all_names or names), registered=names, K={}, R={}, t={}, image_size={})
    for image in model.images.values():
        camera = model.cameras[image.camera_id]
        transform = image.cam_from_world()
        poses['R'][image.name] = transform.rotation.matrix().tolist()
        poses['t'][image.name] = transform.translation.tolist()
        poses['K'][image.name] = camera.calibration_matrix().tolist()
        poses['image_size'][image.name] = [camera.width, camera.height]
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


def export_vggt(path, out, quantile=.5, max_points=50000):
    if not 0 <= quantile < 1: raise ValueError('Confidence quantile must be in [0,1)')
    with np.load(path, allow_pickle=False) as data:
        names = data['image_names'].astype(str).tolist()
        extrinsic, intrinsic = data['extrinsic'], data['intrinsic']
        if extrinsic.shape != (len(names), 3, 4) or intrinsic.shape != (len(names), 3, 3):
            raise ValueError('Expected unbatched extrinsics (N,3,4) and intrinsics (N,3,3)')
        poses = dict(image_names=names, registered=names, K={}, R={}, t={})
        for i, name in enumerate(names):
            poses['K'][name] = intrinsic[i].tolist()
            poses['R'][name], poses['t'][name] = extrinsic[i, :, :3].tolist(), extrinsic[i, :, 3].tolist()
        points = data['world_points'].reshape(-1, 3)
        conf = data['world_points_conf'].reshape(-1)
        finite = np.isfinite(points).all(axis=1) & np.isfinite(conf)
        if not finite.any(): raise ValueError('No finite VGGT points')
        keep = np.flatnonzero(finite & (conf >= np.quantile(conf[finite], quantile)))
        if len(keep) > max_points:
            keep = np.random.default_rng(0).choice(keep, max_points, replace=False)
        colors = None
        if 'images' in data:
            # NPZ stores already preprocessed RGB images as N,H,W,3.
            colors = np.clip(data['images'].reshape(-1, 3)[keep]*255, 0, 255).astype(np.uint8)
        save_ply(points[keep], colors, out/'points.ply')
        write_json(out/'poses.json', poses)
    return poses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--vggt', type=Path); source.add_argument('--colmap', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--confidence-quantile', type=float, default=.5)
    a = p.parse_args()
    if a.vggt: export_vggt(a.vggt, a.out, a.confidence_quantile)
    else:
        import pycolmap
        export_colmap(pycolmap.Reconstruction(a.colmap), a.out)


if __name__ == '__main__': main()
