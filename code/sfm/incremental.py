"""Incremental SfM orchestration. Geometry stays in dedicated testable modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import cheirality_pose, essential_from_F, reprojection_error, triangulate_dlt
from .reconstruction import Camera, Reconstruction


def initialize_two_view(
    name0: str, name1: str, K0: np.ndarray, K1: np.ndarray, keypoints0: np.ndarray, keypoints1: np.ndarray,
    verified_matches: np.ndarray, F: np.ndarray,
) -> Reconstruction:
    """Build a metric-up-to-scale initial reconstruction from one verified pair."""
    p0, p1 = keypoints0[verified_matches[:, 0]], keypoints1[verified_matches[:, 1]]
    R, t = cheirality_pose(essential_from_F(F, K0, K1), p0, p1, K0, K1)
    recon = Reconstruction(cameras={
        0: Camera(0, name0, K0, np.eye(3), np.zeros(3), True),
        1: Camera(1, name1, K1, R, t, True),
    })
    for track_id, (x0, x1) in enumerate(zip(p0, p1)):
        try:
            X = triangulate_dlt(np.asarray([x0, x1]), [np.eye(3), R], [np.zeros(3), t], [K0, K1])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if X[2] > 0 and (R @ X + t)[2] > 0 and reprojection_error(X, x0, K0, np.eye(3), np.zeros(3)) < 2 and reprojection_error(X, x1, K1, R, t) < 2:
            recon.points[track_id] = X
    return recon
