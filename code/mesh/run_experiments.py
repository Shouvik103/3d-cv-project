"""Reproducible one-factor mesh ablations; failures are recorded, never hidden."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from mesh.run_mesh import merge,run,ROOT
from runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/mesh_default.yaml');p.add_argument('--include-stereo',action='store_true');a=p.parse_args()
    cfg=merge(yaml.safe_load((ROOT/'configs/mesh_default.yaml').read_text()),yaml.safe_load(a.config.read_text()))
    variants={'default':{},'voxel_128':{'tsdf':{'voxel_div':128}},'voxel_384':{'tsdf':{'voxel_div':384}},'consistency_off':{'filter':{'consistency':False}},'consistency_on':{'filter':{'consistency':True}},'poisson':{'poisson':{'enabled':True}}}
    if a.include_stereo:variants['sgbm']={'depth_backend':'sgbm'}
    variants['voxel_256']={'tsdf':{'voxel_div':256}}
    rows=[]
    for name,patch in variants.items():
        try:
            result=run(a.scene,a.out/name,merge(cfg,patch));rows.append(dict(name=name,status='complete',full=result['full'],heldout=result['heldout'],seconds=result['timing_seconds']['total']))
        except Exception as e:rows.append(dict(name=name,status='failed',reason=str(e)))
        write_json(a.out/'ablations.json',rows)

if __name__=='__main__':main()
