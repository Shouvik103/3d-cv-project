"""Sequential Depth Anything V2 inference and content/settings keyed offline caches.
Model card: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
License: Apache-2.0 (Small). Predictions are relative inverse depth, not meters.
"""
import os,hashlib,json
from pathlib import Path
import cv2,numpy as np


def cache_key(image_path,config):
    settings={k:config[k] for k in ['model_id','input_long_side']}
    if config.get('revision'):settings['revision']=config['revision']
    h=hashlib.sha256(Path(image_path).read_bytes());h.update(json.dumps(settings,sort_keys=True).encode());h.update(b'mesh-depth-v1');return h.hexdigest()


def save_cache(path,depth,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,depth=np.asarray(depth,dtype='f2'),metadata=np.array(json.dumps(metadata)))


class MonoDepth:
    def __init__(self,config):self.config=config;self.model=None;self.processor=None;self.load_failures=0
    def predict(self,scene,i,cache_only=False):
        cfg=self.config;key=cache_key(scene.image_paths[i],cfg);path=scene.depth_cache/f'{key}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                meta=json.loads(str(d['metadata']));depth=d['depth'].astype('f4')
                if meta['key']!=key or depth.shape!=scene.image(i).shape[:2]:raise ValueError('Depth cache provenance/shape mismatch')
                return depth,dict(cache_hit=True,cache_file=path.name,**meta)
        if cache_only or cfg['model_id']=='analytic-fixture':raise ValueError(f'No matching cached depth for {scene.names[i]} (cache-only requested)')
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')
        import torch
        from transformers import AutoImageProcessor,AutoModelForDepthEstimation
        if self.model is None:
            if self.load_failures>=2:raise RuntimeError('Depth model initialization failed twice; use --backend sgbm or a matching cache')
            self.device=cfg.get('device','auto')
            if self.device=='auto':self.device='mps' if torch.backends.mps.is_available() else 'cpu'
            try:
                self.processor=AutoImageProcessor.from_pretrained(cfg['model_id'],revision=cfg.get('revision'))
                self.model=AutoModelForDepthEstimation.from_pretrained(cfg['model_id'],revision=cfg.get('revision')).to(self.device).eval()
            except (OSError,RuntimeError,ValueError):
                self.load_failures+=1
                raise
        rgb=scene.image(i);h,w=rgb.shape[:2];side=cfg['input_long_side'];scale=side/max(h,w)
        im=cv2.resize(rgb,(max(14,round(w*scale)),max(14,round(h*scale))),interpolation=cv2.INTER_AREA)
        # Disable the processor's default 518px resize; its size divisibility still applies.
        inputs=self.processor(images=im,return_tensors='pt',do_resize=False)
        # ViT patch grids require multiples of 14, so pad/resize explicitly.
        hh,ww=inputs['pixel_values'].shape[-2:];hh=max(14,round(hh/14)*14);ww=max(14,round(ww/14)*14)
        inputs['pixel_values']=torch.nn.functional.interpolate(inputs['pixel_values'],size=(hh,ww),mode='bicubic',align_corners=False)
        with torch.inference_mode():prediction=self.model(**{k:v.to(self.device) for k,v in inputs.items()}).predicted_depth
        depth=torch.nn.functional.interpolate(prediction[:,None],size=(h,w),mode='bicubic',align_corners=False)[0,0].cpu().numpy().astype('f4')
        if not np.isfinite(depth).all():raise ValueError('Nonfinite monocular depth output')
        meta=dict(key=key,model_id=cfg['model_id'],model_revision=getattr(self.model.config,'_commit_hash',None),input_long_side=side,device=self.device,kind='relative_inverse_depth',license='Apache-2.0')
        save_cache(path,depth,meta);return depth,dict(cache_hit=False,cache_file=path.name,**meta)
