"""CPU colored/shaded/wireframe/normal/depth stills and reproducible orbit videos."""
from pathlib import Path
import argparse,sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from mesh.mesh_export import read_mesh
from mesh.rendering import Renderer,orbit_camera,orient_for_display
from runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=180);p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720);p.add_argument('--orbit',action='store_true');p.add_argument('--up',choices=['x','y','z'],default='z');p.add_argument('--poses',type=Path);p.add_argument('--front-image');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    mesh=read_mesh(a.mesh)
    display_transform=orient_for_display(mesh,json.loads(a.poses.read_text()),a.front_image).tolist() if a.poses else None
    if a.up!='z':
        T=np.eye(4);T[:3,:3]=np.array([[1,0,0],[0,0,-1],[0,1,0]]) if a.up=='y' else np.array([[0,0,-1],[0,1,0],[1,0,0]]);mesh.transform(T)
    renderer=Renderer(mesh)
    for view,angle in enumerate([0,2*np.pi/3,4*np.pi/3]):
        K,R,t=orbit_camera(mesh,angle,width=a.width,height=a.height)
        for mode in ['colored','shaded','wireframe','normals']:
            result=renderer.render(K,R,t,a.width,a.height,mode);Image.fromarray(result['color']).save(a.out/f'{mode}_{view}.png')
        np.save(a.out/f'depth_{view}.npy',result['depth'])
        good=result['depth']>0;depth=np.zeros_like(result['depth']);depth[good]=result['depth'][good]/result['depth'][good].max();Image.fromarray(np.rint(depth*65535).astype('uint16')).save(a.out/f'depth_{view}.png')
    if a.orbit:
        if a.frames<2:p.error('--frames must be at least 2')
        import imageio.v2 as imageio
        for mode in ['colored','shaded']:
            with imageio.get_writer(a.out/f'{mode}_orbit.mp4',fps=30,codec='libx264',macro_block_size=2,quality=7) as writer:
                for angle in np.linspace(0,2*np.pi,a.frames,endpoint=False):
                    K,R,t=orbit_camera(mesh,angle,width=a.width,height=a.height);writer.append_data(renderer.render(K,R,t,a.width,a.height,mode)['color'])
    write_json(a.out/'render_manifest.json',dict(mesh=a.mesh.name,width=a.width,height=a.height,orbit=a.orbit,frames=a.frames if a.orbit else 0,background=[245]*3,lighting='fixed camera headlight',up=a.up,display_transform=display_transform,renderer='Open3D CPU RaycastingScene'))

if __name__=='__main__':main()
