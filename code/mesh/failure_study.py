"""Logged analytic failure controls, complementary to real-scene limitations."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from mesh.synthetic import analytic_scene
from mesh.tsdf_fusion import fuse
from mesh.mesh_clean import clean
from mesh.mesh_eval import mesh_statistics
from mesh.rendering import Renderer,orbit_camera
from runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    baseline=analytic_scene(size=112,views=12);cases={'exact_12_views':baseline,'two_views':baseline[:2], 'one_depth_wrong_scale':[dict(f,depth=f['depth']*(1.8 if i==0 else 1)) for i,f in enumerate(baseline)],'all_depth_wrong_scale':[dict(f,depth=f['depth']*1.8) for f in baseline]};rows=[]
    for name,frames in cases.items():
        mesh,_=fuse(frames,.045,.18,10.)
        try:
            mesh,_=clean(mesh,(np.full(3,-5),np.full(3,5)))
            vertices=np.asarray(mesh.vertices);error=abs(np.linalg.norm(vertices,axis=1)-1)
            K,R,t=orbit_camera(mesh);image=Renderer(mesh).render(K,R,t,640,480,'shaded')['color'];Image.fromarray(image).save(a.out/(name+'.png'))
            rows.append(dict(case=name,status='complete',statistics=mesh_statistics(mesh),median_surface_error=float(np.median(error)),p95_surface_error=float(np.quantile(error,.95)),views=len(frames)))
        except ValueError as e:rows.append(dict(case=name,status='failed',reason=str(e),views=len(frames)))
    write_json(a.out/'failure_study.json',dict(scope='Analytic unit sphere, known geometry; deliberate depth corruption after alignment. Not a real-scene accuracy claim.',voxel=.045,seed=0,cases=rows))

if __name__=='__main__':main()
