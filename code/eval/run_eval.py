"""Evaluate a reconstruction against explicit ground truth or a pseudo-reference."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from runtime import manifest, write_json
from sfm.io_utils import load_poses_json, load_ply
from eval.metrics import evaluate_poses, cloud_metrics
from eval.align import transform_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--estimate', type=Path, required=True, help='Directory containing poses.json')
    p.add_argument('--reference', type=Path, required=True, help='Reference poses.json')
    p.add_argument('--reference-kind', choices=['gt', 'colmap'], required=True)
    p.add_argument('--reference-cloud', type=Path)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--config', type=Path, default=Path('configs/eval.yaml'))
    a = p.parse_args(); cfg = yaml.safe_load(a.config.read_text())
    metrics = evaluate_poses(load_poses_json(a.estimate/'poses.json'), load_poses_json(a.reference), cfg['auc_thresholds_deg'])
    if a.reference_cloud:
        s = metrics['alignment']
        points = transform_points(load_ply(a.estimate/'points.ply')[0], s['scale'], np.array(s['rotation']), np.array(s['translation']))
        metrics['pointcloud'] = cloud_metrics(points, load_ply(a.reference_cloud)[0], **cfg['pointcloud'])
    observation_file = a.estimate/'observations.json'
    if observation_file.exists():
        import json
        from sfm.geometry import reprojection_error
        poses = load_poses_json(a.estimate/'poses.json')
        points = load_ply(a.estimate/'points.ply')[0]
        tracks = json.loads(observation_file.read_text())
        if len(points) != len(tracks): raise ValueError('Point/observation export count mismatch')
        errors = [reprojection_error(X, xy, np.array(poses['K'][name]), np.array(poses['R'][name]), np.array(poses['t'][name]))
                  for X, track in zip(points, tracks.values()) for name, xy in track.items()]
        finite = np.asarray(errors)[np.isfinite(errors)]
        metrics['reprojection'] = dict(observations=len(errors), invalid=len(errors)-len(finite),
                                       mean_px=float(np.mean(finite)) if len(finite) else None,
                                       median_px=float(np.median(finite)) if len(finite) else None,
                                       resolution='input images as processed, not original camera resolution')
    summary = a.estimate/'summary.json'
    payload = dict(manifest(cfg), method=a.estimate.name, reference_kind=a.reference_kind, metrics=metrics)
    if summary.exists():
        import json
        payload['run'] = json.loads(summary.read_text())
    write_json(a.out, payload)
    print(metrics)


if __name__ == '__main__': main()
