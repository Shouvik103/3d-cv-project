"""Descriptor matching and from-scratch fundamental-matrix verification."""

from __future__ import annotations

import numpy as np

from .geometry import ransac_fundamental


def match_images(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.8, mutual: bool = True, binary: bool = False, quality: bool = False) -> np.ndarray:
    import cv2

    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2), dtype=int)
    norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=256)) if quality and not binary else cv2.BFMatcher(norm)
    forward = matcher.knnMatch(desc1, desc2, k=2)
    candidates = [(m.distance / max(n.distance,1e-12),m.queryIdx,m.trainIdx) for row in forward if len(row) == 2 for m,n in [row] if m.distance < ratio*n.distance]
    if quality:candidates.sort()
    accepted = [(a,b) for _,a,b in candidates]
    if not mutual:
        return np.asarray(accepted, dtype=int).reshape(-1, 2)
    backward = matcher.match(desc2, desc1)
    reverse = {m.queryIdx: m.trainIdx for m in backward}
    return np.asarray([(a, b) for a, b in accepted if reverse.get(b) == a], dtype=int).reshape(-1, 2)


def verify_pair(kps1: np.ndarray, kps2: np.ndarray, matches: np.ndarray, *, threshold: float = 1.0, iterations: int = 2000, confidence: float = 0.999, seed: int = 0, quality: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if len(matches) < 8:
        raise ValueError("at least eight matches are needed for verification")
    if quality:
        from .robust_matching import verify_ranked
        F,mask=verify_ranked(kps1[matches[:,0]],kps2[matches[:,1]],threshold,iterations,confidence,seed)
    else:F, mask = ransac_fundamental(kps1[matches[:, 0]], kps2[matches[:, 1]], threshold, iterations, confidence=confidence, seed=seed)
    return F, matches[mask]
