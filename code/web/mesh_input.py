"""Prepare mesh photos from original uploads, scaling pixel calibration exactly."""
from pathlib import Path
import json
import shutil

import numpy as np
from PIL import Image,ImageOps
import yaml

from prepare_data import prepare


def upload_masks(job,images,target):
    """Keep supplied PNG alpha at the exact prepared pixel coordinates."""
    job,images,target=Path(job),Path(images),Path(target)
    sources=[p for p in (job/'uploads').iterdir() if p.is_file()]
    has_alpha=False
    for source in sources:
        with Image.open(source) as raw:
            if 'A' in raw.getbands() and raw.getchannel('A').getextrema()[0]<255:has_alpha=True;break
    if not has_alpha:return None
    target.mkdir(parents=True,exist_ok=True)
    for source in sources:
        with Image.open(source) as raw,Image.open(images/(source.stem+'.jpg')) as prepared:
            alpha=ImageOps.exif_transpose(raw).convert('RGBA').getchannel('A').resize(prepared.size,Image.Resampling.NEAREST)
            alpha.save(target/(source.stem+'.png'))
    return target


def prepare_mesh_scene(job, long_side):
    job = Path(job)
    target = job / 'mesh_input'
    prepare(job / 'uploads', target / 'images', long_side)
    if (job/'foreground.json').exists():
        # Spend the depth-pixel budget on the object, using precisely the same
        # original crop as matching. K and observations are scaled below.
        crops={row['image']:row['crop_original'] for row in json.loads((job/'foreground.json').read_text())}
        for upload in (job/'uploads').iterdir():
            name=upload.stem+'.jpg'
            with Image.open(upload) as raw:
                photo=ImageOps.exif_transpose(raw).convert('RGB').crop(crops[name])
                photo.thumbnail((long_side,long_side),Image.Resampling.LANCZOS)
                photo.save(target/'images'/name,quality=95)
    source = job / 'reconstruction'
    poses = json.loads((source / 'poses.json').read_text())
    calibration_path=job/'input/intrinsics.yaml'
    calibration=yaml.safe_load(calibration_path.read_text())['images'] if calibration_path.exists() else {}
    scales, records = {}, {}
    poses['image_size'] = {}
    for name in poses['image_names']:
        with Image.open(job / 'input/images' / name) as image:
            old_w, old_h = image.size
        with Image.open(target / 'images' / name) as image:
            width, height = image.size
        scale = np.array([width / old_w, height / old_h])
        scales[name] = scale
        original=poses['K'].get(name)
        if original is None:original=calibration[name]['K']
        K = np.asarray(original, float)
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
    masks=upload_masks(job,target/'images',target/'masks')
    if (job/'foreground.json').exists():
        masks=target/'masks';masks.mkdir(exist_ok=True)
        for name in poses['image_names']:
            with Image.open(job/'input_masks'/(name+'.png')) as mask,Image.open(target/'images'/name) as photo:
                mask.resize(photo.size,Image.Resampling.NEAREST).save(masks/(name+'.png'))
    (target / 'scene.json').write_text(json.dumps(dict(reconstruction='reconstruction', images='images',**({'masks':'masks'} if masks else {}))))
    return target
