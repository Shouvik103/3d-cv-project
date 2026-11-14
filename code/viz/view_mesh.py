"""Interactive Open3D triangle mesh view, with a window-free screenshot path."""
from pathlib import Path
import argparse,copy,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mesh.mesh_export import read_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--mode',choices=['colored','shaded','wireframe','normals'],default='colored');p.add_argument('--poses',type=Path);p.add_argument('--points',type=Path);p.add_argument('--screenshot',type=Path);a=p.parse_args();mesh=read_mesh(a.mesh)
    if a.screenshot:
        from mesh.rendering import render,orbit_camera
        from PIL import Image
        K,R,t=orbit_camera(mesh);image=render(mesh,K,R,t,640,480,a.mode)['color'];a.screenshot.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(image).save(a.screenshot)
        if a.poses or a.points:print('Headless screenshot renders the mesh; overlays are interactive only.')
        return
    import open3d as o3d,json
    if a.mode=='shaded':mesh.paint_uniform_color([.65]*3)
    elif a.mode=='normals':mesh.vertex_colors=o3d.utility.Vector3dVector((np.asarray(mesh.vertex_normals)+1)/2)
    objects=[o3d.geometry.LineSet.create_from_triangle_mesh(mesh) if a.mode=='wireframe' else mesh]
    if a.points:objects.append(o3d.io.read_point_cloud(str(a.points)))
    if a.poses:
        poses=json.loads(a.poses.read_text());scale=np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent())*.06
        for n in poses['registered']:
            K=np.array(poses['K'][n]);ext=np.eye(4);ext[:3,:3]=poses['R'][n];ext[:3,3]=poses['t'][n];w,h=poses.get('image_size',{}).get(n,[round(K[0,2]*2),round(K[1,2]*2)])
            objects.append(o3d.geometry.LineSet.create_camera_visualization(w,h,K,ext,scale))
    print('Left drag: orbit; scroll: zoom; Shift+drag: pan; R: reset; H: help. Close window to exit.')
    o3d.visualization.draw_geometries(objects,mesh_show_back_face=True,window_name=a.mesh.name)

if __name__=='__main__':main()
