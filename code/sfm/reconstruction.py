"""Data structures shared by reconstruction, export, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class Camera:
    image_id: int
    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    registered: bool = False

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Reconstruction:
    cameras: Dict[int, Camera] = field(default_factory=dict)
    points: Dict[int, np.ndarray] = field(default_factory=dict)
    tracks: Dict[int, Dict[int, int]] = field(default_factory=dict)
    colors: Dict[int, np.ndarray] = field(default_factory=dict)

    def registered_ids(self) -> list[int]:
        return [idx for idx, camera in self.cameras.items() if camera.registered]
