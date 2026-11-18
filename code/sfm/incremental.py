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
    def solve(indices):
        selected=[cams[i] for i in indices]
        return triangulate_dlt(obs[indices],[c.R for c in selected],[c.t for c in selected],[c.K for c in selected])
    def consensus(X):
        if not np.isfinite(X).all():return []
        return [i for i,(xy,c) in enumerate(zip(obs,cams)) if (c.R@X+c.t)[2]>0 and reprojection_error(X,xy,c.K,c.R,c.t)<=config['max_reproj_px']]
    def parallax(X,indices):
        return max((triangulation_angle_deg(X,cams[a].R,cams[a].t,cams[b].R,cams[b].t) for a,b in combinations(indices,2)),default=0)
    try:
        X=solve(list(range(len(ids))));valid=consensus(X)
        if len(valid)==len(ids) and parallax(X,valid)>=config['min_angle_deg']:return X
    except (ValueError,np.linalg.LinAlgError):pass
    if not config.get('robust',False) or len(ids)<3:return None
    # Bounded triangulation consensus salvages a track with a bad observation.
    centers=[-c.R.T@c.t for c in cams]
    pairs=sorted(combinations(range(len(ids)),2),key=lambda ij:-np.linalg.norm(centers[ij[0]]-centers[ij[1]]))[:8]
    best=None
    for pair in pairs:
        try:X=solve(list(pair));valid=consensus(X)
        except (ValueError,np.linalg.LinAlgError):continue
        if len(valid)<max(2,int(np.ceil(.6*len(ids)))) or parallax(X,valid)<config['min_angle_deg']:continue
        error=np.median([reprojection_error(X,obs[i],cams[i].K,cams[i].R,cams[i].t) for i in valid])
        if best is None or (len(valid),-error)>best[0]:best=((len(valid),-error),X,valid)
    if best is None:return None
    _,X,valid=best
    try:
        refined=solve(valid);refined_valid=consensus(refined)
        if len(refined_valid)>=len(valid) and parallax(refined,refined_valid)>=config['min_angle_deg']:X,valid=refined,refined_valid
    except (ValueError,np.linalg.LinAlgError):pass
    for i in set(range(len(ids)))-set(valid):track.pop(ids[i])
    return X


def choose_initial_pair(pairs, matrices, homography_ratios, keypoints, cameras, config,tracks=None,triangulation=None):
    candidates = sorted(pairs, key=lambda pair: len(pairs[pair]), reverse=True)[:config['top_k_pairs']]
    if tracks is not None:
        strongest={}
        for pair in sorted(pairs,key=lambda pair:-len(pairs[pair])):
            for view in pair:strongest.setdefault(view,pair)
        candidates=list(dict.fromkeys(candidates+list(strongest.values())))
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
        support=len(pairs[(i,j)])
        coverage={}
        if tracks is not None:
            # Score only observations that survived track conflict resolution and
            # can actually seed triangulation, not merely pairwise F inliers.
            support=0
            for track in tracks.values():
                if i not in track or j not in track:continue
                xy=np.array([keypoints[i][track[i]],keypoints[j][track[j]]])
                X=triangulate_dlt(xy,[np.eye(3),R],[np.zeros(3),t],[a.K,b.K])
                if X[2]<=0 or (R@X+t)[2]<=0:continue
                if max(reprojection_error(X,xy[0],a.K,np.eye(3),np.zeros(3)),reprojection_error(X,xy[1],b.K,R,t))>triangulation['max_reproj_px']:continue
                if triangulation_angle_deg(X,np.eye(3),np.zeros(3),R,t)<triangulation['min_angle_deg']:continue
                support+=1
                for view in track:coverage[view]=coverage.get(view,0)+1
            if support<20:continue
        score = support * min(float(np.median(angles)), 15)
        if tracks is not None:
            # Prefer a seed connected to many future PnP views over an isolated
            # highly textured cluster (e.g. only the roof of a toy car).
            reachable=sum(count>=15 for count in coverage.values())
            score=reachable**2*np.sqrt(support)*min(float(np.median(angles)),10)
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


def filter_points(recon, keypoints, threshold,prune_observations=False):
    removed = []
    for k, X in recon.points.items():
        records=[(i,reprojection_error(X,keypoints[i][kp],c.K,c.R,c.t) if not prune_observations or (c.R@X+c.t)[2]>0 else float('inf'))
                 for i,kp in recon.tracks[k].items() if (c:=recon.cameras[i]).registered]
        errors=[error for _,error in records]
        if prune_observations and sum(error<=threshold for error in errors)>=2:
            for i,error in records:
                if error>threshold:recon.tracks[k].pop(i)
            continue
        if not errors or max(errors) > threshold:
            removed.append(k)
    for k in removed:
        del recon.points[k]
    return len(removed)
