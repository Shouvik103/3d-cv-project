"""Keep COLMAP's native threading runtime out of the learned-depth worker."""
from pathlib import Path
import argparse,json,resource,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import write_json


def main():
    from baselines.run_colmap import run
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--object-quality',action='store_true');p.add_argument('--maximum-quality',action='store_true');a=p.parse_args()
    run(a.images,a.out,object_quality=a.object_quality,maximum_quality=a.maximum_quality,progress=lambda phase,done,total,message:print('WEB_PROGRESS '+json.dumps([phase,done,total,message]),flush=True))
    summary=json.loads((a.out/'summary.json').read_text())
    summary['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    write_json(a.out/'summary.json',summary)


if __name__=='__main__':main()
