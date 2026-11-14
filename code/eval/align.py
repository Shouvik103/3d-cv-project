"""Similarity alignment of estimated camera centers to a reference."""
import numpy as np


def umeyama(source, target):
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError('Alignment needs at least three paired 3D points')
    xc, yc = x-x.mean(0), y-y.mean(0)
    if np.linalg.matrix_rank(xc) < 2 or np.linalg.matrix_rank(yc) < 2:
        raise ValueError('Camera centers must not be collinear')
    U, d, Vt = np.linalg.svd(yc.T @ xc / len(x))
    signs = np.array([1., 1., np.linalg.det(U @ Vt)])
    Q = (U * signs) @ Vt
    scale = float(np.dot(d, signs) / np.mean(np.sum(xc**2, axis=1)))
    offset = y.mean(0) - scale * Q @ x.mean(0)
    return scale, Q, offset


def transform_points(points, scale, rotation, translation):
    return scale * np.asarray(points) @ rotation.T + translation
