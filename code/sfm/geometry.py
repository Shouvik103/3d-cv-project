"""From-scratch multi-view geometry primitives using the OpenCV pose convention."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


EPS = 1e-12


def _as_h(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    return points if points.shape[1] == 3 else np.c_[points, np.ones(len(points))]


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley-normalize pixel coordinates; returns homogeneous points and T."""
    x = _as_h(points)
    xy = x[:, :2] / x[:, 2:3]
    centroid = xy.mean(axis=0)
    mean_distance = np.linalg.norm(xy - centroid, axis=1).mean()
    scale = math.sqrt(2.0) / max(mean_distance, EPS)
    T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1.]])
    return (T @ x.T).T, T


def fundamental_8pt(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Estimate a rank-2 fundamental matrix with the normalized 8-point method."""
    if len(points1) != len(points2) or len(points1) < 8:
        raise ValueError("at least eight paired points are required")
    x1, T1 = normalize_points(points1)
    x2, T2 = normalize_points(points2)
    u1, v1 = x1[:, 0], x1[:, 1]
    u2, v2 = x2[:, 0], x2[:, 1]
    A = np.column_stack([u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones(len(x1))])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, s, Vt = np.linalg.svd(F)
    s[-1] = 0.0
    F = U @ np.diag(s) @ Vt
    F = T2.T @ F @ T1
    norm = np.linalg.norm(F)
    return F / norm if norm > EPS else F


def sampson_distance(F: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    x1, x2 = _as_h(points1), _as_h(points2)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numer = np.sum(x2 * Fx1, axis=1) ** 2
    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    return numer / np.maximum(denom, EPS)


def ransac_fundamental(
    points1: np.ndarray, points2: np.ndarray, thresh: float = 1.0,
    iters: int = 2000, confidence: float = 0.999, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """8-point RANSAC with deterministic sampling and adaptive iteration count."""
    n = len(points1)
    if n < 8:
        raise ValueError("at least 8 tentative matches are required")
    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    best_F: np.ndarray | None = None
    cap, iteration = iters, 0
    threshold_sq = thresh * thresh
    while iteration < min(iters, cap):
        indices = rng.choice(n, size=8, replace=False)
        try:
            candidate = fundamental_8pt(points1[indices], points2[indices])
        except np.linalg.LinAlgError:
            iteration += 1
            continue
        mask = sampson_distance(candidate, points1, points2) <= threshold_sq
        if mask.sum() > best_mask.sum():
            best_F, best_mask = candidate, mask
            ratio = mask.mean()
            if ratio > 0:
                failure_prob = max(EPS, 1.0 - ratio**8)
                cap = min(cap, max(iteration + 1, int(math.ceil(math.log(1-confidence) / math.log(failure_prob)))))
        iteration += 1
    if best_F is None or best_mask.sum() < 8:
        raise RuntimeError("RANSAC failed to find a non-degenerate fundamental matrix")
    F = fundamental_8pt(points1[best_mask], points2[best_mask])
    mask = sampson_distance(F, points1, points2) <= threshold_sq
    return F, mask


def essential_from_F(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    E = np.asarray(K2).T @ F @ np.asarray(K1)
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0:
        Vt *= -1
    return U @ np.diag([1., 1., 0.]) @ Vt


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1] *= -1
    W = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotations = [U @ W @ Vt, U @ W.T @ Vt]
    rotations = [R if np.linalg.det(R) > 0 else -R for R in rotations]
    t = U[:, 2] / max(np.linalg.norm(U[:, 2]), EPS)
    return [(R, sign * t) for R in rotations for sign in (1., -1.)]


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.asarray(K) @ np.c_[np.asarray(R), np.asarray(t).reshape(3)]


def triangulate_dlt(
    observations: np.ndarray, rotations: list[np.ndarray], translations: list[np.ndarray], Ks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Triangulate one point from two or more observations with n-view DLT."""
    if len(observations) < 2 or len(observations) != len(rotations) or len(rotations) != len(translations):
        raise ValueError("two or more matching observations, poses, and translations are required")
    rows: list[np.ndarray] = []
    for idx, (xy, R, t) in enumerate(zip(observations, rotations, translations)):
        x = np.array([xy[0], xy[1], 1.0])
        if Ks is not None:
            x = np.linalg.inv(Ks[idx]) @ x
        x /= x[2]
        P = np.c_[R, np.asarray(t).reshape(3)]
        rows.extend((x[0] * P[2] - P[0], x[1] * P[2] - P[1]))
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    Xh = Vt[-1]
    if abs(Xh[3]) < EPS:
        raise ValueError("triangulation returned a point at infinity")
    return Xh[:3] / Xh[3]


def cheirality_pose(E: np.ndarray, points1: np.ndarray, points2: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select the E decomposition that places the most points in front of both cameras."""
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for R, t in decompose_essential(E):
        valid = 0
        for x1, x2 in zip(points1, points2):
            try:
                X = triangulate_dlt(np.asarray([x1, x2]), [np.eye(3), R], [np.zeros(3), t], [K1, K2])
            except (ValueError, np.linalg.LinAlgError):
                continue
            if X[2] > 0 and (R @ X + t)[2] > 0:
                valid += 1
        if best is None or valid > best[0]:
            best = (valid, R, t)
    if best is None or best[0] == 0:
        raise RuntimeError("essential-matrix cheirality selection failed")
    return best[1], best[2]


def reprojection_error(X: np.ndarray, xy: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    q = np.asarray(K) @ (np.asarray(R) @ np.asarray(X) + np.asarray(t).reshape(3))
    if q[2] <= EPS:
        return float("inf")
    return float(np.linalg.norm(q[:2] / q[2] - np.asarray(xy)))


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return -np.asarray(R).T @ np.asarray(t).reshape(3)


def triangulation_angle_deg(X: np.ndarray, R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> float:
    a, b = np.asarray(X) - camera_center(R1, t1), np.asarray(X) - camera_center(R2, t2)
    cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), EPS)
    return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
