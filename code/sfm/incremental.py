"""Initialization, triangulation, filtering, and local/global bundle adjustment."""
from itertools import combinations
import logging
import numpy as np
from .bundle_adjust import bundle_adjust
from .geometry import cheirality_pose, essential_from_F, reprojection_error, triangulate_dlt, triangulation_angle_deg

logger = logging.getLogger('sfm')


def triangulate_track(recon, track, keypoints, config):
    ids = [i for i in track if recon.cameras[i].registered]
    if len(ids) < 2:
        return None
    cams = [recon.cameras[i] for i in ids]
    obs = np.array([keypoints[i][track[i]] for i in ids])
    try:
        X = triangulate_dlt(obs, [c.R for c in cams], [c.t for c in cams], [c.K for c in cams])
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(X).all() or any((c.R @ X + c.t)[2] <= 0 for c in cams):
        return None
    if max(reprojection_error(X, xy, c.K, c.R, c.t) for xy, c in zip(obs, cams)) > config['max_reproj_px']:
        return None
    angle = max(triangulation_angle_deg(X, a.R, a.t, b.R, b.t) for a, b in combinations(cams, 2))
    return X if angle >= config['min_angle_deg'] else None


def choose_initial_pair(pairs, matrices, homography_ratios, keypoints, cameras, config):
    candidates = sorted(pairs, key=lambda pair: len(pairs[pair]), reverse=True)[:config['top_k_pairs']]
    best = None
    for i, j in candidates:
        if homography_ratios[(i, j)] > config['max_hf_ratio']:
            continue
        matches = pairs[(i, j)]
        # A bounded deterministic subset is enough for cheirality and parallax scoring.
        matches = matches[np.linspace(0, len(matches)-1, min(200, len(matches)), dtype=int)]
        p0, p1 = keypoints[i][matches[:, 0]], keypoints[j][matches[:, 1]]
        a, b = cameras[i], cameras[j]
        try:
            R, t = cheirality_pose(essential_from_F(matrices[(i, j)], a.K, b.K), p0, p1, a.K, b.K)
            angles = []
            for x, y in zip(p0, p1):
                X = triangulate_dlt(np.array([x, y]), [np.eye(3), R], [np.zeros(3), t], [a.K, b.K])
                if X[2] > 0 and (R @ X + t)[2] > 0:
                    angles.append(triangulation_angle_deg(X, np.eye(3), np.zeros(3), R, t))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if len(angles) < 0.7 * len(matches) or np.median(angles) < config['min_median_angle_deg']:
            continue
        score = len(pairs[(i, j)]) * min(float(np.median(angles)), 15)
        if best is None or score > best[0]:
            best = (score, i, j, R, t)
    if best is None:
        raise RuntimeError('No initialization pair passes homography/parallax checks; use more textured views with translation')
    return best[1:]


def run_ba(recon, keypoints, config, anchors, window=None):
    ids = recon.registered_ids()
    point_ids = sorted(recon.points, key=lambda k: len(recon.tracks[k]), reverse=True)
    if window is not None:
        point_ids = [k for k in point_ids if any(i in window for i in recon.tracks[k])]
    point_ids = point_ids[:config['max_points_global']]
    if not point_ids:
        return
    camera_slot = {i: slot for slot, i in enumerate(ids)}
    ci, pi, obs, Ks = [], [], [], []
    for slot, k in enumerate(point_ids):
        for i, kp in recon.tracks[k].items():
            if i in camera_slot:
                ci.append(camera_slot[i]); pi.append(slot)
                obs.append(keypoints[i][kp]); Ks.append(recon.cameras[i].K)
    fixed = tuple(camera_slot[i] for i in ids if i == anchors[0] or (window is not None and i not in window))
    result = bundle_adjust(np.array([recon.cameras[i].R for i in ids]), np.array([recon.cameras[i].t for i in ids]),
                           np.array([recon.points[k] for k in point_ids]), np.array(ci), np.array(pi), np.array(obs), np.array(Ks),
                           fixed_camera_ids=fixed, scale_camera_ids=tuple(camera_slot[i] for i in anchors), loss=config['loss'], f_scale=config['f_scale'],
                           max_nfev=config['max_nfev_local'] if window is not None else config['max_nfev'],
                           refine_focal=bool(config.get('refine_focal',False) and window is None and len(ids)>=4))
    if not np.isfinite(result.final_rms):
        raise RuntimeError('Bundle adjustment produced nonfinite residuals')
    for slot, i in enumerate(ids):
        recon.cameras[i].R, recon.cameras[i].t = result.rotations[slot], result.translations[slot]
    for slot, k in enumerate(point_ids):
        recon.points[k] = result.points[slot]
    if result.focal_scale!=1.:
        # Shared camera/lens correction also applies to unregistered images.
        # Modify K in place so the PnP intrinsics lookup sees the same update.
        for camera in recon.cameras.values():
            camera.K[0,0]*=result.focal_scale;camera.K[1,1]*=result.focal_scale
        logger.info('Global BA focal multiplier %.6f (principal points fixed)',result.focal_scale)
    logger.info('BA RMS %.3f -> %.3f px, %d evaluations', result.initial_rms, result.final_rms, result.nfev)


def local_window(recon, image_id, count):
    neighbors = {i: 0 for i in recon.registered_ids() if i != image_id}
    for k in recon.points:
        if image_id in recon.tracks[k]:
            for i in recon.tracks[k]:
                if i in neighbors:
                    neighbors[i] += 1
    return {image_id, *sorted(neighbors, key=neighbors.get, reverse=True)[:count]}


def filter_points(recon, keypoints, threshold):
    removed = []
    for k, X in recon.points.items():
        errors = [reprojection_error(X, keypoints[i][kp], c.K, c.R, c.t)
                  for i, kp in recon.tracks[k].items() if (c := recon.cameras[i]).registered]
        if not errors or max(errors) > threshold:
            removed.append(k)
    for k in removed:
        del recon.points[k]
    return len(removed)
