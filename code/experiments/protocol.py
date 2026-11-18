"""Freeze matched input subsets and calibration before running any method."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import yaml

from prepare_data import prepare
from runtime import manifest, write_json

ROOT = Path(__file__).resolve().parents[2]
ORDER = ['E1','E2','E3','E4','E5','E6','E7','M4','M5']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def select_uniform(names, count):
    names = sorted(names)
    if not 3 <= count <= len(names):
        raise ValueError('A benchmark needs at least three available views')
    return [names[int(i)] for i in np.linspace(0,len(names)-1,count,dtype=int)]


def prepare_scene(key, scene, config, output):
    output = Path(output)
    images = ROOT / scene['images']
    names = select_uniform([p.name for p in images.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}],scene['views'])
    inputs = dict(scene=key, selected_images=names, source_images=scene['images'],
                  source_sha256={n:sha256(images/n) for n in names},
                  intrinsics_sha256=sha256(ROOT/scene['intrinsics']),
                  reference_sha256=sha256(ROOT/scene['reference']),
                  max_side=config['image_max_side'], seed=config['seed'])
    previous = output / 'inputs.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if old['inputs'] != inputs:
            raise ValueError(f'Frozen inputs changed for {key}; use a new benchmark output directory')
        for name, digest in old['prepared_sha256'].items():
            if sha256(output/'images'/name) != digest:
                raise ValueError(f'Prepared benchmark photo changed: {name}')
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Partially prepared dataset {output}; choose a new output directory')
    raw = output / 'source'
    raw.mkdir(parents=True)
    for name in names:
        shutil.copy2(images/name, raw/name)
    prepare(raw, output/'images', config['image_max_side'], ROOT/scene['intrinsics'])
    reference = json.loads((ROOT/scene['reference']).read_text())
    mapping = {n:Path(n).stem+'.jpg' for n in names}
    reference['image_names'] = list(mapping.values())
    reference['registered'] = [mapping[n] for n in names if n in reference['registered']]
    # Reference cameras are used for pose evaluation only; preserve their native
    # calibration rather than pretending these pixels match our prepared JPEGs.
    for field in ['K','R','t','image_size','distortion']:
        if field in reference:
            reference[field] = {mapping[n]:reference[field][n] for n in names if n in reference[field]}
    write_json(output/'reference_poses.json',reference)
    record = dict(manifest(config), inputs=inputs, reference_kind=scene['reference_kind'],
                  notes=scene['notes'], reference_camera_frame='Native reference cameras, used only for pose evaluation',
                  prepared_sha256={mapping[n]:sha256(output/'images'/mapping[n]) for n in names})
    write_json(previous,record)
    return record


def update_ledger(output, experiment, status, **details):
    output = Path(output)
    path = output/'experiment_status.json'
    ledger = json.loads(path.read_text()) if path.exists() else dict(order=ORDER,experiments={k:dict(status='pending') for k in ORDER})
    ledger['experiments'][experiment] = dict(status=status,**details)
    write_json(path,ledger)
    return ledger
