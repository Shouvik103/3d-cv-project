"""Run VGGT only inside its isolated NumPy-compatible environment."""
from pathlib import Path
import argparse,json,logging,resource,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import write_json,manifest


def main():
    from baselines.vggt_infer import infer
    from baselines.convert_to_common import export_vggt
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--masks',type=Path);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');a.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    meta=infer(a.images,a.out/'raw.npz')
    if a.masks:
        import numpy as np
        from vggt.utils.load_fn import load_and_preprocess_images
        with np.load(a.out/'raw.npz',allow_pickle=False) as archive:arrays={key:archive[key] for key in archive.files}
        paths=[str(a.masks/(Path(name).stem+'.png')) for name in arrays['image_names'].astype(str)]
        mask=load_and_preprocess_images(paths)[:,0].numpy()>.5
        if mask.shape!=arrays['images'].shape[:3]:raise ValueError('VGGT alpha mask preprocessing shape mismatch')
        arrays['valid_masks']=mask;np.savez_compressed(a.out/'raw.npz',**arrays)
        meta['foreground_mask']='Original upload alpha, transformed with official VGGT preprocessing; predictions unchanged'
        write_json(a.out/'raw.json',meta)
    poses=export_vggt(a.out/'raw.npz',a.out,max_points=1000000)
    with (a.out/'points.ply').open() as stream:
        for line in stream:
            if line.startswith('element vertex '):count=int(line.split()[-1]);break
    write_json(a.out/'summary.json',dict(manifest(),method='vggt',registered=len(poses['registered']),total_images=len(poses['image_names']),
        points=count,runtime_seconds=time.perf_counter()-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        registration_scope='Predicted cameras, not feature-verified registration',cloud_scope='Confidence-filtered point-head export, capped at 1 million points; raw dense arrays retained in raw.npz',vggt=meta))


if __name__=='__main__':main()
