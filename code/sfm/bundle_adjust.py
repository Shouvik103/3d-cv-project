"""Sparse, robust bundle adjustment with an explicitly fixed initial baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BAResult:
    rotations: np.ndarray
    translations: np.ndarray
    points: np.ndarray
    initial_rms: float
    final_rms: float
    nfev: int
    focal_scale: float = 1.0


def _rvecs_from_rotations(rotations: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotations).as_rotvec()


def _rotations_from_rvecs(rvecs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rvecs).as_matrix()


def _project(points: np.ndarray, rotations: np.ndarray, translations: np.ndarray, Ks: np.ndarray) -> np.ndarray:
    camera_points = np.einsum("nij,nj->ni", rotations, points) + translations
    pixels = np.einsum("nij,nj->ni", Ks, camera_points)
    return pixels[:, :2] / np.where(np.abs(pixels[:, 2:3]) < 1e-12, 1e-12, pixels[:, 2:3])


def bundle_adjust(
    rotations: np.ndarray,
    translations: np.ndarray,
    points: np.ndarray,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    observations: np.ndarray,
    intrinsics: np.ndarray,
    *,
    fixed_camera_ids: tuple[int, ...] = (0,),
    scale_camera_ids: tuple[int, int] = (0, 1),
    loss: str = "huber",
    f_scale: float = 1.0,
    max_nfev: int = 100,
    refine_focal: bool = False,
) -> BAResult:
    """Optimize camera poses and points with scipy's sparse trust-region solver.

    The first pose is fixed; one camera-center distance constrains scale.
    This lets the second pose refine instead of freezing its two-view errors.
    """
    if refine_focal:
        from .focal_adjust import bundle_adjust_focal
        return bundle_adjust_focal(rotations,translations,points,camera_indices,point_indices,observations,intrinsics,
                                   fixed_camera_ids,scale_camera_ids,loss,f_scale,max_nfev)
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    R0 = np.asarray(rotations, dtype=float).copy()
    t0 = np.asarray(translations, dtype=float).reshape(-1, 3).copy()
    X0 = np.asarray(points, dtype=float).copy()
    camera_indices, point_indices = np.asarray(camera_indices, dtype=int), np.asarray(point_indices, dtype=int)
    observations, intrinsics = np.asarray(observations, dtype=float), np.asarray(intrinsics, dtype=float)
    n_cameras, n_points = len(R0), len(X0)
    a, b = scale_camera_ids
    def baseline(Rs, ts):
        centers = -np.einsum('nji,nj->ni', Rs[[a, b]], ts[[a, b]])
        return np.linalg.norm(centers[0]-centers[1])
    baseline0 = baseline(R0, t0)
    if baseline0 <= 1e-12:
        raise ValueError('Scale anchors must have nonzero baseline')
    scale_weight = float(np.median(intrinsics[:, 0, 0])) / baseline0
    free_ids = np.asarray([i for i in range(n_cameras) if i not in fixed_camera_ids], dtype=int)
    free_slot = {camera: slot for slot, camera in enumerate(free_ids)}
    rvec0 = _rvecs_from_rotations(R0)
    initial = _project(X0[point_indices], R0[camera_indices], t0[camera_indices], intrinsics) - observations

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rvecs, translations = rvec0.copy(), t0.copy()
        count = len(free_ids)
        rvecs[free_ids] = params[:3 * count].reshape(count, 3)
        translations[free_ids] = params[3 * count:6 * count].reshape(count, 3)
        pts = params[6 * count:].reshape(n_points, 3)
        return _rotations_from_rvecs(rvecs), translations, pts

    def residuals(params: np.ndarray) -> np.ndarray:
        Rs, ts, X = unpack(params)
        pixels = (_project(X[point_indices], Rs[camera_indices], ts[camera_indices], intrinsics) - observations).ravel()
        return np.r_[pixels, scale_weight * (baseline(Rs, ts)-baseline0)]

    n_free = len(free_ids)
    sparsity = lil_matrix((2 * len(observations) + 1, 6 * n_free + 3 * n_points), dtype=int)
    rows = np.arange(len(observations))
    for camera, slot in free_slot.items():
        mask = camera_indices == camera
        for column in range(6):
            offset = 3 * slot + column if column < 3 else 3 * len(free_ids) + 3 * slot + column - 3
            sparsity[2 * rows[mask], offset] = 1
            sparsity[2 * rows[mask] + 1, offset] = 1
    for column in range(3):
        sparsity[2 * rows, 6 * n_free + 3 * point_indices + column] = 1
        sparsity[2 * rows + 1, 6 * n_free + 3 * point_indices + column] = 1
    for camera in (a, b):
        if camera in free_slot:
            slot = free_slot[camera]
            sparsity[-1, 3*slot:3*slot+3] = 1
            sparsity[-1, 3*n_free+3*slot:3*n_free+3*slot+3] = 1
    x0 = np.r_[rvec0[free_ids].ravel(), t0[free_ids].ravel(), X0.ravel()]
    result = least_squares(residuals, x0, jac_sparsity=sparsity.tocsr(), method="trf", tr_solver="lsmr", x_scale="jac", loss=loss, f_scale=f_scale, max_nfev=max_nfev)
    Rs, ts, X = unpack(result.x)
    final = residuals(result.x)[:-1]
    return BAResult(Rs, ts, X, float(np.sqrt(np.mean(initial**2))), float(np.sqrt(np.mean(final**2))), result.nfev)
