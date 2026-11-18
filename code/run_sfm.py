"""Reconstruct an unordered image set: python code/run_sfm.py --help."""
from __future__ import annotations
import argparse
from copy import deepcopy
from itertools import combinations
import logging
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import yaml
from runtime import manifest, write_json
from sfm.features import cached_features
from sfm.geometry import reprojection_error
from sfm.incremental import choose_initial_pair, triangulate_track, run_ba, local_window, filter_points
from sfm.io_utils import save_ply, save_poses_json
from sfm.matching import match_images, verify_pair
from sfm.reconstruction import Camera, Reconstruction
from sfm.tracks import build_tracks

logger = logging.getLogger('sfm')


def load_intrinsics(images, names, color_images):
    candidate = images.parent / 'intrinsics.yaml'
    records = yaml.safe_load(candidate.read_text()).get('images', {}) if candidate.exists() else {}
    result = {}
    for name, image in zip(names, color_images):
        h, w = image.shape[:2]
        if candidate.exists():
            if name not in records:
                raise ValueError(f'Missing intrinsics for {name}')
            K = np.asarray(records[name]['K'], dtype=float)
            if 'image_size' in records[name] and records[name]['image_size'] != [w, h]:
                raise ValueError(f'Intrinsics image size does not match {name}')
        else:
            f = 1.2 * max(w, h)
            K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]])
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f'Invalid intrinsics for {name}')
        result[name] = K
    return result


def run_sfm(images: Path, output: Path, config: dict, max_images=None, progress=None):
    started = time.perf_counter()
    config = deepcopy(config)
    def report(phase, done, total, message):
        if progress is not None:
            progress(phase, done, total, message)
    cv2.setRNGSeed(config['seed']); cv2.setNumThreads(1)
    names = sorted(p.name for p in images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if max_images is not None:
        if max_images < 2:
            raise ValueError('--max-images must be at least 2')
        names = names[:max_images]
    if len(names) < 2:
        raise ValueError('SfM needs at least two images')
    output.mkdir(parents=True, exist_ok=True)
    color_images = [cv2.imread(str(images / n)) for n in names]
    if any(im is None for im in color_images):
        raise ValueError('One or more images could not be decoded')
    Ks = load_intrinsics(images, names, color_images)
    initial_focal={n:float(Ks[n][0,0]) for n in names} if config['ba'].get('refine_focal',False) else None
    fcfg, mcfg = config['features'], config['matching']
    keypoints, descriptors = [], []
    for feature_index, (name, image) in enumerate(zip(names, color_images), 1):
        mask_path=images.parent/'masks'/(name+'.png')
        mask=cv2.imread(str(mask_path),0) if mask_path.exists() else None
        xy, desc = cached_features(image, images/name, fcfg, images.parent/'cache',mask)
        keypoints.append(xy); descriptors.append(desc)
        logger.info('Features %s: %d', name, len(xy))
        report('features', feature_index, len(names), f'Extracting features · {feature_index}/{len(names)} images')
    pairs, matrices, hf = {}, {}, {}
    if mcfg.get('bounded_pairs',False):
        from sfm.pair_selection import select_pairs
        pair_list=select_pairs(color_images,features=descriptors)
    else:pair_list=list(combinations(range(len(names)),2))
    pair_count = len(pair_list)
    for pair_index, (i, j) in enumerate(pair_list):
        report('matching', pair_index, pair_count, f'Matching views · {pair_index:,}/{pair_count:,} pairs checked')
        tentative = match_images(descriptors[i], descriptors[j], mcfg['ratio'], mcfg['mutual'], fcfg['detector'] == 'orb')
        try:
            F, matches = verify_pair(keypoints[i], keypoints[j], tentative, threshold=mcfg['ransac_thresh_px'],
                                     iterations=mcfg['ransac_iters'], confidence=mcfg['ransac_conf'], seed=config['seed']+i*1000+j)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if len(matches) < mcfg['min_inliers']:
            continue
        _, mask = cv2.findHomography(keypoints[i][tentative[:, 0]], keypoints[j][tentative[:, 1]], cv2.RANSAC, mcfg['ransac_thresh_px'])
        pairs[(i, j)], matrices[(i, j)] = matches, F
        hf[(i, j)] = float(mask.sum()/len(matches)) if mask is not None else 0.
        logger.info('Pair %s/%s: %d inliers, H/F %.2f', names[i], names[j], len(matches), hf[(i, j)])
    report('matching', pair_count, pair_count, f'Matching views · {pair_count:,}/{pair_count:,} pairs checked')
    if not pairs:
        raise RuntimeError('No image pair passed geometric verification')
    recon = Reconstruction(cameras={i: Camera(i, n, Ks[n], np.eye(3), np.zeros(3), False) for i, n in enumerate(names)})
    recon.tracks = build_tracks(pairs,conflict_aware=config.get('conflict_aware_tracks',False))
    report('initialization', 0, 1, 'Initializing the camera geometry')
    i0, i1, R, t = choose_initial_pair(pairs, matrices, hf, keypoints, recon.cameras, config['init'],
                                     recon.tracks if config.get('conflict_aware_tracks') else None,config['triangulation'])
    recon.cameras[i0].registered = True
    recon.cameras[i1].R, recon.cameras[i1].t, recon.cameras[i1].registered = R, t, True
    anchors = (i0, i1)
    ba = config['ba']; mode = ba.get('mode', 'local_global')
    history = []

    def triangulate_new():
        added = 0
        for k, track in recon.tracks.items():
            if k not in recon.points:
                X = triangulate_track(recon, track, keypoints, config['triangulation'])
                if X is not None:
                    recon.points[k] = X; added += 1
        return added

    def snapshot():
        errors = [reprojection_error(X, keypoints[i][kp], c.K, c.R, c.t)
                  for k, X in recon.points.items() for i, kp in recon.tracks[k].items() if (c := recon.cameras[i]).registered]
        item = dict(registered=len(recon.registered_ids()), points=len(recon.points),
                    median_reprojection_px=float(np.median(errors)) if errors else None, elapsed_seconds=time.perf_counter()-started)
        history.append(item); logger.info('State: %s', item)
        if config.get('export', {}).get('save_growth_snapshots', False):
            dest = output / 'growth' / f'{len(history):03d}'
            save_poses_json(recon, dest / 'poses.json')
            save_ply(np.array(list(recon.points.values())), None, dest / 'points.ply')

    triangulate_new()
    if len(recon.points) < config['registration']['min_inliers']:
        raise RuntimeError('Initialization yielded too few usable 3D points')
    if mode != 'none':
        run_ba(recon, keypoints, ba, anchors)
    snapshot()
    report('initialization', 1, 1, 'Initial camera pair reconstructed')
    report('registration', 2, len(names), f'Estimating camera poses · 2/{len(names)} registered')
    while True:
        candidates = []
        for i in range(len(names)):
            if recon.cameras[i].registered:
                continue
            keys = [k for k in recon.points if i in recon.tracks[k]]
            candidates.append((len(keys), i, keys))
        logger.info('Registration candidates: %s', [(names[i], count) for count, i, _ in sorted(candidates, reverse=True)])
        success = False
        for count, i, keys in sorted(candidates, reverse=True):
            if count < config['registration']['min_inliers']:
                continue
            X = np.array([recon.points[k] for k in keys], dtype=np.float64)
            xy = np.array([keypoints[i][recon.tracks[k][i]] for k in keys], dtype=np.float64)
            reg = config['registration']
            ok, rv, tv, inliers = cv2.solvePnPRansac(X, xy, Ks[names[i]], None, reprojectionError=reg['pnp_reproj_px'], iterationsCount=reg['pnp_iters'], flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < reg['min_inliers']:
                logger.info('Deferring %s after PnP failure', names[i]); continue
            sel = inliers[:, 0]
            rv, tv = cv2.solvePnPRefineLM(X[sel], xy[sel], Ks[names[i]], None, rv, tv)
            recon.cameras[i].R, recon.cameras[i].t = cv2.Rodrigues(rv)[0], tv.ravel()
            recon.cameras[i].registered = True
            # Drop rejected observations before they can bias BA or triangulation.
            projected = cv2.projectPoints(X, rv, tv, Ks[names[i]], None)[0].reshape(-1, 2)
            valid = (np.linalg.norm(projected-xy, axis=1) <= reg['pnp_reproj_px']) & ((recon.cameras[i].R @ X.T).T[:, 2] + tv.ravel()[2] > 0)
            for k, good in zip(keys, valid):
                if not good:
                    recon.tracks[k].pop(i)
            triangulate_new()
            if mode != 'none':
                run_ba(recon, keypoints, ba, anchors, local_window(recon, i, ba['local_neighbors']))
                if mode == 'local_global' and len(recon.registered_ids()) % ba['global_every'] == 0:
                    run_ba(recon, keypoints, ba, anchors)
                filter_points(recon, keypoints, config['filter']['final_reproj_px'],config['filter'].get('prune_observations',False))
            snapshot()
            count_registered = len(recon.registered_ids())
            report('registration', count_registered, len(names), f'Estimating camera poses · {count_registered}/{len(names)} registered')
            success = True; break
        if not success:
            break
    report('optimization', 0, 1, 'Refining camera poses and 3D points')
    if mode == 'local_global':
        run_ba(recon, keypoints, ba, anchors)
    removed = filter_points(recon, keypoints, config['filter']['final_reproj_px'],config['filter'].get('prune_observations',False))
    if mode == 'local_global' and removed:
        run_ba(recon, keypoints, ba, anchors)
        filter_points(recon, keypoints, config['filter']['final_reproj_px'],config['filter'].get('prune_observations',False))
    if not recon.points:
        raise RuntimeError('No 3D points survived final filtering')
    snapshot()
    report('optimization', 1, 1, 'Camera poses and points refined')
    colors = []
    for k in recon.points:
        i, kp = next((i, kp) for i, kp in recon.tracks[k].items() if recon.cameras[i].registered)
        x, y = np.round(keypoints[i][kp]).astype(int); im = color_images[i]
        colors.append(im[np.clip(y, 0, im.shape[0]-1), np.clip(x, 0, im.shape[1]-1)][::-1])
    save_poses_json(recon, output / 'poses.json')
    save_ply(np.array(list(recon.points.values())), np.array(colors), output / 'points.ply')
    observations = {str(k): {names[i]: keypoints[i][kp].tolist() for i, kp in recon.tracks[k].items() if recon.cameras[i].registered} for k in recon.points}
    write_json(output / 'observations.json', observations)
    summary = dict(manifest(config), **history[-1], total_images=len(names), runtime_seconds=time.perf_counter()-started,
                   init_pair=[names[i0], names[i1]], image_names=names, history=history,
                   mean_track_length=float(np.mean([len(observations[str(k)]) for k in recon.points])) if recon.points else 0.)
    if initial_focal:
        summary['focal_refinement']=dict(initial_fx=initial_focal,final_fx={c.name:float(c.K[0,0]) for c in recon.cameras.values()},
                                          scope='Shared fx/fy multiplier in global BA with at least four registered views; principal points fixed; per-call multiplier bounded [0.5,2]')
    write_json(output / 'summary.json', summary)
    report('sfm_export', 1, 1, 'Sparse reconstruction saved')
    return recon


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/sfm_default.yaml'))
    p.add_argument('--max-images', '--max_images', type=int)
    p.add_argument('--detector', choices=['sift', 'rootsift', 'orb'])
    p.add_argument('--ba', choices=['none', 'local', 'local_global'])
    p.add_argument('--no-ba', '--no_ba', action='store_true')
    a = p.parse_args(); cfg = yaml.safe_load(a.config.read_text())
    if a.detector: cfg['features']['detector'] = a.detector
    if a.ba: cfg['ba']['mode'] = a.ba
    if a.no_ba: cfg['ba']['mode'] = 'none'
    a.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(a.out / 'run.log')])
    run_sfm(a.images, a.out, cfg, a.max_images)


if __name__ == '__main__':
    main()
