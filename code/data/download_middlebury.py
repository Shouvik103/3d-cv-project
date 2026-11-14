"""Download the official sparse-ring archives and convert their calibration."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import yaml
from runtime import write_json

BASE = 'https://vision.middlebury.edu/mview/data/data/'


def parse_cameras(path):
    lines = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    expected = int(lines.pop(0)[0]) if len(lines[0]) == 1 else len(lines)
    cameras = {}
    for row in lines:
        values = np.array(row[1:], float)
        if len(values) != 21: raise ValueError('Expected filename, K(9), R(9), t(3)')
        cameras[row[0]] = (values[:9].reshape(3, 3), values[9:18].reshape(3, 3), values[18:])
    if len(cameras) != expected: raise ValueError('Camera count mismatch')
    return cameras


def download(scene, out):
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f'{scene}.zip'
    url = BASE + f'{scene}.zip'
    if not archive.exists():
        print('Downloading', url, flush=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=120) as response, archive.with_suffix('.zip.part').open('wb') as dest:
            shutil.copyfileobj(response, dest)
        archive.with_suffix('.zip.part').replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (out / member.filename).resolve()
            if not target.is_relative_to(out.resolve()): raise ValueError('Unsafe archive path')
        z.extractall(out)
    root = out / scene
    params = next(root.glob('*_par.txt'))
    cameras = parse_cameras(params)
    names = sorted(cameras)
    payload = dict(image_names=names, registered=names, K={}, R={}, t={}, image_size={})
    records = {}
    for name in names:
        K, R, t = cameras[name]
        size = list(Image.open(root/name).size)
        payload['K'][name], payload['R'][name], payload['t'][name] = K.tolist(), R.tolist(), t.tolist()
        payload['image_size'][name] = size
        records[name] = dict(K=K.tolist(), image_size=size, source='middlebury_calibrated')
    write_json(root/'gt_poses.json', payload)
    (root/'intrinsics.yaml').write_text(yaml.safe_dump(dict(images=records)))
    write_json(root/'provenance.json', dict(url=url, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), source='Middlebury MVS', calibration='K[R|t], world-to-camera'))
    print('Downloaded calibrated scene:', root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=['templeSparseRing', 'dinoSparseRing'], default='templeSparseRing')
    p.add_argument('--out', type=Path, default=Path('data/middlebury'))
    a = p.parse_args(); download(a.scene, a.out)


if __name__ == '__main__': main()
