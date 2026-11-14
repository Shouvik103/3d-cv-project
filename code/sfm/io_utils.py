"""Portable data formats used at every method boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .reconstruction import Reconstruction


def save_poses_json(reconstruction: Reconstruction, path: str | Path) -> None:
    path = Path(path)
    cameras = sorted(reconstruction.cameras.values(), key=lambda camera: camera.name)
    payload = {
        "image_names": [camera.name for camera in cameras],
        "K": {camera.name: np.asarray(camera.K).tolist() for camera in cameras},
        "R": {camera.name: np.asarray(camera.R).tolist() for camera in cameras if camera.registered},
        "t": {camera.name: np.asarray(camera.t).reshape(3).tolist() for camera in cameras if camera.registered},
        "registered": [camera.name for camera in cameras if camera.registered],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_poses_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_ply(points: np.ndarray, colors: np.ndarray | None, path: str | Path) -> None:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    colors = np.full((len(points), 3), 200, dtype=np.uint8) if colors is None else np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("points and colors must have equal length")
    header = "\n".join(["ply", "format ascii 1.0", f"element vertex {len(points)}", "property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"])
    rows = "\n".join(f"{x:.8f} {y:.8f} {z:.8f} {r} {g} {b}" for (x, y, z), (r, g, b) in zip(points, colors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n{rows}\n")


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the ASCII XYZ/RGB PLY format emitted by this project."""
    with Path(path).open() as stream:
        if stream.readline().strip() != 'ply' or stream.readline().strip() != 'format ascii 1.0':
            raise ValueError('Expected an ASCII PLY')
        count = None
        for line in stream:
            if line.startswith('element vertex '): count = int(line.split()[-1])
            if line.strip() == 'end_header': break
        if count is None: raise ValueError('Missing vertex count')
        rows = [list(map(float, stream.readline().split())) for _ in range(count)]
    data = np.asarray(rows).reshape(count, -1) if count else np.empty((0, 6))
    return data[:, :3], data[:, 3:6].astype(np.uint8)
