"""Prepare mesh photos from original uploads, scaling pixel calibration exactly."""
from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image
import yaml

from prepare_data import prepare


def prepare_mesh_scene(job, long_side):
    job = Path(job)
    target = job / 'mesh_input'
    prepare(job / 'uploads', target / 'images', long_side)
    source = job / 'reconstruction'
    poses = json.loads((source / 'poses.json').read_text())
    scales, records = {}, {}
    poses['image_size'] = {}
    for name in poses['image_names']:
        with Image.open(job / 'input/images' / name) as image:
            old_w, old_h = image.size
        with Image.open(target / 'images' / name) as image:
            width, height = image.size
        scale = np.array([width / old_w, height / old_h])
        scales[name] = scale
        K = np.asarray(poses['K'][name], float)
        K[:2] *= scale[:, None]
        poses['K'][name] = K.tolist()
        poses['image_size'][name] = [width, height]
        records[name] = dict(K=K.tolist(), image_size=[width,height], source='scaled_from_reconstructed_camera')
    observations = json.loads((source / 'observations.json').read_text())
    for track in observations.values():
        for name, uv in track.items():
            track[name] = (np.asarray(uv) * scales[name]).tolist()
    reconstruction = target / 'reconstruction'
    reconstruction.mkdir(exist_ok=True)
    (reconstruction / 'poses.json').write_text(json.dumps(poses, indent=2))
    (reconstruction / 'observations.json').write_text(json.dumps(observations))
    shutil.copyfile(source / 'points.ply', reconstruction / 'points.ply')
    (target / 'intrinsics.yaml').write_text(yaml.safe_dump(dict(images=records)))
    (target / 'scene.json').write_text(json.dumps(dict(reconstruction='reconstruction', images='images')))
    return target
