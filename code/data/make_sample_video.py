"""Make a synthetic video fixture: held views, blur, and an unrelated final shot.

This is an integration-test input, not a recording of a real object in motion.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from runtime import write_json

ROOT = Path(__file__).resolve().parents[2]


def make_video(output, scene=ROOT/'sample_data/scene_small'):
    if output.suffix.lower() != '.avi':
        raise ValueError('Use .avi for the portable MJPEG fixture')
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Choose an unused fixture filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'MJPG'), 12., (800, 600))
    if not writer.isOpened():
        raise RuntimeError('MJPEG video encoding is unavailable')
    mapping, blurred = [], []
    try:
        for view, hold in enumerate([12, 24, 10, 28, 12, 20, 16, 10]):
            name = f'view_{view:02d}.png'
            image = cv2.imread(str(scene/'images'/name))
            if image is None or image.shape[:2] != (600, 800):
                raise ValueError('Fixture requires the original 800x600 generated sample images')
            if view:
                blur = cv2.GaussianBlur(image, (31, 31), 8)
                for _ in range(3):
                    blurred.append(len(mapping)); mapping.append(name); writer.write(blur)
            for _ in range(hold):
                mapping.append(name); writer.write(image)
        unrelated = np.random.default_rng(9).integers(0, 256, (600, 800, 3), dtype=np.uint8)
        for _ in range(18):
            mapping.append(None); writer.write(unrelated)
    finally:
        writer.release()
    write_json(output.with_suffix('.json'), dict(kind='synthetic_video_fixture', fps=12.,
               frame_to_source_image=mapping, blurred_indices=blurred,
               unrelated_shot_start_frame=153, source_poses=str(scene/'gt_poses.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    make_video(a.out)
