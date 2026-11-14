"""VGGT inference for a CUDA/Colab environment; no training or local weights."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime import manifest, write_json


def main():
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--images', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-images', type=int, default=20); a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Run this adapter on a CUDA GPU, for example Colab T4')
    paths = sorted(p for p in a.images.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})[:a.max_images]
    if len(paths) < 2: raise ValueError('At least two images are required')
    torch.manual_seed(0)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT.from_pretrained('facebook/VGGT-1B').to('cuda').eval()
    images = load_and_preprocess_images([str(p) for p in paths]).to('cuda')
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype): predictions = model(images)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-started
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
    def cpu(t): return t.squeeze(0).float().cpu().numpy()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, image_names=np.array([p.name for p in paths]), extrinsic=cpu(extrinsic), intrinsic=cpu(intrinsic),
                        world_points=cpu(predictions['world_points']), world_points_conf=cpu(predictions['world_points_conf']),
                        images=images.float().cpu().permute(0, 2, 3, 1).numpy())
    write_json(a.out.with_suffix('.json'), dict(manifest(), runtime_seconds=elapsed, timing_scope='GPU forward pass only; excludes loading and export',
                                              peak_gpu_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(), image_names=[p.name for p in paths]))


if __name__ == '__main__': main()
