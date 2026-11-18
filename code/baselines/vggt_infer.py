"""Pinned VGGT inference on CUDA, Apple MPS or CPU, without training."""
import argparse
import contextlib
import gc
import importlib.metadata
import logging
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from runtime import manifest,write_json

MODEL_REVISION='860abec7937da0a4c03c41d3c269c366e82abdf9'
CODE_REVISION='a288dd0f14786c93483e45524328726ab7b1b4ce'


def infer(images_path,out,device='auto',max_images=None,model_revision=MODEL_REVISION):
    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    if device=='auto':device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; select --device mps/cpu or use Colab')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Apple MPS is unavailable')
    paths=sorted(p for p in Path(images_path).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'})
    if max_images is not None:paths=paths[:max_images]
    if len(paths)<2:raise ValueError('At least two images are required')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);torch.set_num_threads(4)
    logging.info('VGGT %d images, device=%s, pinned checkpoint=%s',len(paths),device,model_revision)
    download_start=time.perf_counter()
    checkpoint=hf_hub_download('facebook/VGGT-1B','model.safetensors',revision=model_revision)
    download_seconds=time.perf_counter()-download_start
    loading_start=time.perf_counter()
    # Meta construction avoids allocating a second five-gigabyte FP32 model.
    with init_empty_weights(include_buffers=False):
        model=VGGT(enable_track=False)
    weights=load_file(checkpoint,device='cpu')
    incompatible=model.load_state_dict(weights,strict=False,assign=True)
    if incompatible.missing_keys or any(not key.startswith('track_head.') for key in incompatible.unexpected_keys):
        raise ValueError(f'Unexpected checkpoint layout: {incompatible}')
    del weights;gc.collect()
    # These nonpersistent normalization buffers are not checkpoint tensors.
    model.aggregator._resnet_mean=torch.tensor([.485,.456,.406]).view(1,1,3,1,1)
    model.aggregator._resnet_std=torch.tensor([.229,.224,.225]).view(1,1,3,1,1)
    if any(t.is_meta for t in list(model.parameters())+list(model.buffers())):
        raise ValueError('An uninitialized checkpoint tensor remains')
    dtype=torch.float16 if device=='mps' else torch.float32
    model.aggregator.to(device=device,dtype=dtype)
    for head in [model.camera_head,model.depth_head,model.point_head]:head.to(device=device,dtype=torch.float32)
    model.eval()
    images=load_and_preprocess_images([str(p) for p in paths]).to(device)
    loading_seconds=time.perf_counter()-loading_start
    def synchronize():
        if device=='cuda':torch.cuda.synchronize()
        elif device=='mps':torch.mps.synchronize()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    peaks=dict(allocated_bytes=0,driver_bytes=0);stop=threading.Event()
    def sample_memory():
        while not stop.wait(.1):
            peaks['allocated_bytes']=max(peaks['allocated_bytes'],torch.mps.current_allocated_memory())
            peaks['driver_bytes']=max(peaks['driver_bytes'],torch.mps.driver_allocated_memory())
    sampler=threading.Thread(target=sample_memory,daemon=True) if device=='mps' else None
    if sampler:sampler.start()
    synchronize();started=time.perf_counter()
    try:
        if device=='cuda':
            amp_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
            amp=torch.autocast('cuda',dtype=amp_dtype)
        elif device=='mps':amp=torch.autocast('mps',dtype=torch.float16)
        else:amp=contextlib.nullcontext()
        with torch.inference_mode():
            logging.info('VGGT aggregator: input shape %s',tuple(images.shape))
            with amp:tokens,start_idx=model.aggregator(images[None])
            logging.info('VGGT prediction heads (FP32, one frame per dense-head chunk)')
            tokens=[None if tensor is None else tensor.float() for tensor in tokens]
            pose=model.camera_head(tokens)[-1]
            extrinsic,intrinsic=pose_encoding_to_extri_intri(pose,images.shape[-2:])
            depth,depth_conf=model.depth_head(tokens,images=images[None],patch_start_idx=start_idx,frames_chunk_size=1)
            world,world_conf=model.point_head(tokens,images=images[None],patch_start_idx=start_idx,frames_chunk_size=1)
        synchronize();elapsed=time.perf_counter()-started
    finally:
        stop.set()
        if sampler:sampler.join(timeout=2)
    def cpu(tensor):return tensor.squeeze(0).float().cpu().numpy()
    arrays=dict(image_names=np.array([p.name for p in paths]),extrinsic=cpu(extrinsic),intrinsic=cpu(intrinsic),
                world_points=cpu(world),world_points_conf=cpu(world_conf),images=images.float().cpu().permute(0,2,3,1).numpy(),
                depth=cpu(depth),depth_conf=cpu(depth_conf))
    if any(not np.isfinite(value).all() for key,value in arrays.items() if key!='image_names'):
        raise RuntimeError('VGGT produced nonfinite predictions; this attempt cannot be used as a result')
    np.savez_compressed(out,**arrays)
    metadata=dict(manifest(),device=device,model_id='facebook/VGGT-1B',model_revision=model_revision,
                  repository_revision=CODE_REVISION,vggt_version=importlib.metadata.version('vggt'),
                  runtime_seconds=elapsed,loading_preprocessing_seconds=loading_seconds,
                  checkpoint_resolution_seconds=download_seconds,
                  timing_scope='Synchronized aggregator plus all prediction heads; excludes model loading and NPZ export',
                  precision='FP16 aggregator and FP32 heads' if device=='mps' else ('CUDA autocast aggregator and FP32 heads' if device=='cuda' else 'FP32'),
                  preprocessing='Unmodified official crop preprocessing, width 518 px',input_shape=list(images.shape),
                  unused_track_head_disabled=True,dense_head_chunk_frames=1,image_names=[p.name for p in paths],
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                  sampled_mps_peak=peaks if device=='mps' else None)
    write_json(out.with_suffix('.json'),metadata)
    logging.info('VGGT predictions saved: %s (%.2fs forward)',out,elapsed)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-images',type=int);parser.add_argument('--device',choices=['auto','cuda','mps','cpu'],default='auto')
    parser.add_argument('--model-revision',default=MODEL_REVISION)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    infer(args.images,args.out,args.device,args.max_images,args.model_revision)


if __name__=='__main__':main()
