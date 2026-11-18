"""Pose and density-matched point-cloud metrics in the common convention."""
from itertools import combinations
import numpy as np
from scipy.spatial import cKDTree
from scipy.integrate import trapezoid
from .align import umeyama, transform_points


def angle_deg(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1)))) if denom > 1e-12 else float('inf')


def rotation_error(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def pose_auc(errors, threshold):
    """Include infinite failures in the recall denominator."""
    errors = np.sort(np.asarray(errors, float))
    if len(errors) == 0:
        return 0.
    recall = np.arange(1, len(errors)+1) / len(errors)
    valid = errors <= threshold
    x = np.r_[0., errors[valid]]; y = np.r_[0., recall[valid]]
    # CDF stays constant after the last accepted error; do not interpolate toward infinity.
    x = np.r_[x, threshold]; y = np.r_[y, y[-1]]
    return float(trapezoid(y, x) / threshold)


def centers(poses, names):
    return np.array([-np.asarray(poses['R'][n]).T @ np.asarray(poses['t'][n]) for n in names])


def evaluate_poses(estimate, reference, thresholds=(5, 10, 30), allow_partial=False):
    names = reference['image_names']
    shared = sorted(set(estimate['registered']) & set(reference['registered']))
    alignment_error = None
    try:
        scale, Q, b = umeyama(centers(estimate, shared), centers(reference, shared))
        target = centers(reference, shared)
        residuals = np.linalg.norm(transform_points(centers(estimate, shared), scale, Q, b)-target, axis=1)
        alignment = dict(scale=scale, rotation=Q.tolist(), translation=b.tolist())
        rmse = float(np.sqrt(np.mean(residuals**2)))
    except ValueError as exc:
        if not allow_partial:raise
        alignment_error = str(exc)
        alignment = None
        rmse = None
    errors, rotations, translations = [], [], []
    shared_set = set(shared)
    undefined_reference_pairs = 0
    for i, j in combinations(names, 2):
        if i in reference['registered'] and j in reference['registered']:
            Ri, Rj = np.array(reference['R'][i]), np.array(reference['R'][j])
            baseline = np.array(reference['t'][j]) - Rj @ Ri.T @ np.array(reference['t'][i])
            if np.linalg.norm(baseline) < 1e-8:
                undefined_reference_pairs += 1
                continue  # Translation direction is undefined for co-located GT cameras.
        if i not in shared_set or j not in shared_set:
            errors.append(float('inf')); continue
        rel = []
        for poses in (estimate, reference):
            Ri, Rj = np.array(poses['R'][i]), np.array(poses['R'][j])
            ti, tj = np.array(poses['t'][i]), np.array(poses['t'][j])
            R = Rj @ Ri.T; rel.append((R, tj-R @ ti))
        re = rotation_error(rel[0][0] @ rel[1][0].T)
        te = angle_deg(rel[0][1], rel[1][1])
        rotations.append(re); translations.append(te); errors.append(max(re, te))
    extent = float(np.linalg.norm(np.ptp(centers(reference, reference['registered']), axis=0)))
    metrics = dict(registered=len(shared), total_images=len(names), registration_rate=len(shared)/len(names), ate_rmse=rmse,
                   ate_percent_extent=100*rmse/extent if extent and rmse is not None else None,
                   median_rotation_deg=float(np.median(rotations)) if rotations else None,
                   median_translation_deg=float(np.median(translations)) if translations and np.isfinite(translations).all() else None,
                   auc={str(t): pose_auc(errors, t) for t in thresholds}, pose_pairs=len(errors),
                   undefined_reference_pairs=undefined_reference_pairs,
                   alignment=alignment, alignment_error=alignment_error, compared_images=shared)
    return metrics


def cloud_metrics(estimate, reference, max_points=50000, tau_frac=0.01, seed=0):
    a, b = np.asarray(estimate), np.asarray(reference)
    if not len(a) or not len(b):
        raise ValueError('Point clouds must be nonempty')
    rng = np.random.default_rng(seed); count = min(len(a), len(b), max_points)
    a = a[rng.choice(len(a), count, replace=False)]; b = b[rng.choice(len(b), count, replace=False)]
    tau = float(tau_frac * np.linalg.norm(np.ptp(b, axis=0)))
    da = cKDTree(b).query(a)[0]; db = cKDTree(a).query(b)[0]
    precision, recall = float(np.mean(da <= tau)), float(np.mean(db <= tau))
    return dict(chamfer=float((da.mean()+db.mean())/2), precision=precision, recall=recall,
                fscore=2*precision*recall/(precision+recall) if precision+recall else 0., tau=tau, sampled_points=count)
