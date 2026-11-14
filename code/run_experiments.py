"""Run reproducible BA/detector/view-count ablations on an explicit scene."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import yaml
from runtime import write_json, manifest
from run_sfm import run_sfm
from sfm.io_utils import load_poses_json
from eval.metrics import evaluate_poses


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--config', type=Path, default=Path('configs/sfm_default.yaml'))
    p.add_argument('--ablation', choices=['ba', 'detector', 'views'], required=True)
    p.add_argument('--views', type=int, nargs='+', default=[5, 10, 20, 30]); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(); cfg = yaml.safe_load(a.config.read_text()); cfg['seed'] = a.seed
    values = {'ba': ['none', 'local', 'local_global'], 'detector': ['sift', 'rootsift', 'orb'], 'views': a.views}[a.ablation]
    names = sorted(p.name for p in (a.scene/'images').iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    reference = load_poses_json(a.scene/'gt_poses.json')
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    for value in values:
        config = deepcopy(cfg); images = a.scene/'images'; out = a.out/f'{a.ablation}_{value}_seed{a.seed}'
        if (out/'summary.json').exists():
            print('Already completed:', out, flush=True); continue
        selected = names
        if a.ablation == 'ba': config['ba']['mode'] = value
        elif a.ablation == 'detector': config['features']['detector'] = value
        elif value > len(names): continue
        else:
            selected = sorted(np.random.default_rng(a.seed).choice(names, value, replace=False).tolist())
            images = out/'input'/'images'; images.mkdir(parents=True, exist_ok=True)
            for name in selected: shutil.copy2(a.scene/'images'/name, images/name)
            shutil.copy2(a.scene/'intrinsics.yaml', images.parent/'intrinsics.yaml')
        try:
            run_sfm(images, out, config)
            # View-count subsets use the identical reference subset; failures within it remain in AUC.
            ref = deepcopy(reference); ref['image_names'] = selected; ref['registered'] = [n for n in selected if n in ref['registered']]
            metrics = evaluate_poses(load_poses_json(out/'poses.json'), ref)
            write_json(out/'evaluation.json', dict(manifest(config), method=out.name, reference_kind='gt', metrics=metrics, run=json.loads((out/'summary.json').read_text())))
        except (ValueError, RuntimeError) as exc:
            write_json(out/'failure.json', dict(manifest(config), error=str(exc), selected_images=selected))
            print('Failed:', value, exc, flush=True)


if __name__ == '__main__': main()
