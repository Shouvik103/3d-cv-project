"""Estimate a checkerboard camera calibration at the supplied image resolution."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--images', type=Path, required=True)
    p.add_argument('--board', nargs=2, type=int, default=[9, 6], metavar=('COLS', 'ROWS'))
    p.add_argument('--square-size', type=float, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.square_size <= 0: raise ValueError('Square size must be positive')
    obj = np.zeros((a.board[0]*a.board[1], 3), np.float32); obj[:, :2] = np.mgrid[:a.board[0], :a.board[1]].T.reshape(-1, 2)*a.square_size
    objects, pixels, names = [], [], []; size = None
    for path in sorted(a.images.iterdir()):
        image = cv2.imread(str(path))
        if image is None: continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); current = gray.shape[::-1]
        if size is not None and current != size: raise ValueError('All calibration images must have identical dimensions')
        size = current; ok, corners = cv2.findChessboardCorners(gray, tuple(a.board))
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, .001))
            objects.append(obj); pixels.append(corners); names.append(path.name)
    if len(objects) < 5: raise ValueError('At least five successful checkerboard detections required')
    rms, K, dist, Rs, ts = cv2.calibrateCamera(objects, pixels, size, None, None)
    errors = {n: float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, dist)[0]-xy)**2, axis=2)))) for n, o, xy, r, t in zip(names, objects, pixels, Rs, ts)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(dict(K=K.tolist(), distortion=dist.ravel().tolist(), image_size=list(size), rms_px=float(rms), per_image_rms_px=errors)))
    print(f'Calibration RMS: {rms:.3f} px over {len(names)} images')


if __name__ == '__main__': main()
