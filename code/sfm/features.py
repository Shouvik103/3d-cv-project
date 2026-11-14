"""Feature extraction with portable, compressed caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_features(image: np.ndarray, detector: str = "sift", n_features: int = 4000, rootsift: bool = False) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detector in {"sift", "rootsift"}:
        engine = cv2.SIFT_create(nfeatures=n_features)
    elif detector == "orb":
        engine = cv2.ORB_create(nfeatures=n_features)
    else:
        raise ValueError(f"unsupported detector: {detector}")
    keypoints, descriptors = engine.detectAndCompute(gray, None)
    xy = np.array([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)
    descriptors = np.empty((0, 128 if detector != "orb" else 32), dtype=np.float32) if descriptors is None else descriptors
    if rootsift or detector == "rootsift":
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
