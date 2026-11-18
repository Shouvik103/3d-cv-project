"""Feature extraction with portable, compressed caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_features(image: np.ndarray, detector: str = "sift", n_features: int = 4000, rootsift: bool = False, contrast_threshold: float = 0.04,mask=None) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detector in {"sift", "rootsift"}:
        engine = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold)
    elif detector == "orb":
        engine = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError(f"unsupported detector: {detector}")
    keypoints, descriptors = engine.detectAndCompute(gray,mask)
    xy = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
    descriptors = np.empty((0, 128 if detector != "orb" else 32), dtype=np.float32) if descriptors is None else descriptors
    if detector != "orb" and (rootsift or detector == "rootsift"):
        descriptors = descriptors.astype(np.float32)
        descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
        descriptors = np.sqrt(descriptors)
    return xy, descriptors


def save_feature_cache(path: Path, names: list[str], features: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    payload: dict[str, np.ndarray] = {"names": np.asarray(names)}
    for index, name in enumerate(names):
        payload[f"kps_{index}"] = features[name][0]
        payload[f"desc_{index}"] = features[name][1]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)



def cached_features(image, path, settings, cache_dir,mask=None):
    """Content-addressed features; edits to images/settings invalidate the cache."""
    import hashlib
    import json
    import cv2
    identity = hashlib.sha256(path.read_bytes() + json.dumps(settings, sort_keys=True).encode() + cv2.__version__.encode()+(mask.tobytes() if mask is not None else b'')).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f'{identity}.npz'
    if cached.exists():
        with np.load(cached, allow_pickle=False) as data:
            return data['xy'], data['descriptors']
    xy, descriptors = extract_features(image, settings['detector'], settings['n_features'], settings.get('rootsift', False), settings.get('contrast_threshold', .04),mask)
    np.savez_compressed(cached, xy=xy, descriptors=descriptors)
    return xy, descriptors
