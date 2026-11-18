"""One isolated benchmark method; emit per-process performance and failures."""
from pathlib import Path
import argparse
import json
import logging
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from runtime import manifest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['ours','colmap','vggt'],required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--variant-config',type=Path)
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    root=Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'run.log')])
    started=time.perf_counter()
    status='failed';error=None
    try:
        if args.method=='ours':
            from run_sfm import run_sfm
            cfg=yaml.safe_load(args.variant_config.read_text()) if args.variant_config else yaml.safe_load((root/config['sfm_config']).read_text())
            cfg['seed']=config['seed'];cfg['export']['save_growth_snapshots']=True
            run_sfm(args.scene/'images',args.out,cfg,
                    progress=lambda phase,done,total,message: logging.info('%s',message) if done in (0,total) or (done % max(1,total//10)==0) else None)
        elif args.method=='colmap':
            from baselines.run_colmap import run
            run(args.scene/'images',args.out)
        else:
            from baselines.vggt_infer import infer
            cfg=config['vggt']
            infer(args.scene/'images',args.out/'raw.npz',device=cfg.get('device','auto'),
                  model_revision=cfg['model_revision'])
            from baselines.convert_to_common import export_vggt
            export_vggt(args.out/'raw.npz',args.out)
        status='completed'
    except Exception as exc:
        error=str(exc);logging.exception('Benchmark method failed')
    finally:
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
        write_json(args.out/'performance.json',dict(manifest(config),method=args.method,status=status,
                   error=error,wall_seconds=time.perf_counter()-started,peak_process_rss_mb=peak,
                   timing_scope='End-to-end method process, including model loading/export; excludes environment installation and checkpoint download'))
    return 0 if status=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
