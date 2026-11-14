"""CPU ray casting; no OpenGL/window required. Pixel centers are (u+.5,v+.5)."""
import numpy as np
from .synthetic import look_at


class Renderer:
    def __init__(self,mesh):
        import open3d as o3d
        self.mesh=mesh;mesh.compute_vertex_normals();self.scene=o3d.t.geometry.RaycastingScene(nthreads=4);self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    def render(self,K,R,t,W,H,mode='colored',background=(245,245,245)):
        import open3d as o3d
        ext=np.eye(4);ext[:3,:3]=R;ext[:3,3]=t
        rays=self.scene.create_rays_pinhole(o3d.core.Tensor(np.asarray(K,dtype='f8')),o3d.core.Tensor(ext),W,H)
        result=self.scene.cast_rays(rays,nthreads=4);hit=np.isfinite(result['t_hit'].numpy());ids=result['primitive_ids'].numpy()[hit];uv=result['primitive_uvs'].numpy()[hit];weights=np.column_stack([1-uv.sum(1),uv])
        triangles=np.asarray(self.mesh.triangles)[ids];normals=np.zeros((H,W,3),'f4');n=(np.asarray(self.mesh.vertex_normals)[triangles]*weights[:,:,None]).sum(1);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-10);normals[hit]=n
        colors=np.asarray(self.mesh.vertex_colors) if self.mesh.has_vertex_colors() else np.full((len(self.mesh.vertices),3),.65)
        rgb=(colors[triangles]*weights[:,:,None]).sum(1)
        directions=rays.numpy()[:,:,3:];unit=directions/np.maximum(np.linalg.norm(directions,axis=2,keepdims=True),1e-10)
        if mode=='shaded':rgb=np.repeat((.18+.82*np.maximum(0,(n*-unit[hit]).sum(1)))[:,None],3,axis=1)
        elif mode=='normals':rgb=(n+1)/2
        elif mode=='wireframe':
            edge=weights.min(1)<.025;rgb[:]=.9;rgb[edge]=.12
        image=np.full((H,W,3),background,dtype='u1');image[hit]=np.rint(np.clip(rgb,0,1)*255).astype('u1')
        depth=np.zeros((H,W),'f4')
        # Open3D rays are not unit length. t_hit multiplies the actual direction.
        depth[hit]=result['t_hit'].numpy()[hit]*(directions[hit]@np.asarray(R)[2])
        return dict(color=image,depth=depth,normals=normals,mask=hit)


def render(mesh,K,R,t,W,H,mode='colored',background=(245,245,245)):
    return Renderer(mesh).render(K,R,t,W,H,mode,background)


def orbit_camera(mesh,angle=0,elevation=.35,width=640,height=480):
    bounds=np.asarray(mesh.get_axis_aligned_bounding_box().get_extent());center=np.asarray(mesh.get_axis_aligned_bounding_box().get_center())
    f=max(width,height)*1.1
    radius=float(np.linalg.norm(bounds))*.5/np.sin(np.arctan(height/(2*f)))*1.08
    # Display z up; use the native coordinate frame unless --up is specified by caller.
    eye=center+radius*np.array([np.cos(angle)*np.cos(elevation),np.sin(angle)*np.cos(elevation),np.sin(elevation)])
    R,t=look_at(eye,center);K=np.array([[f,0,width/2],[0,f,height/2],[0,0,1.]])
    return K,R,t


def orient_for_display(mesh,poses,front_image=None):
    """Rigid display transform only; exported reconstruction remains untouched."""
    names=poses['registered'];Rs=np.array([poses['R'][n] for n in names]);_,_,V=np.linalg.svd(Rs[:,0,:],full_matrices=False);up=V[-1]
    if up@(-Rs[:,1,:]).mean(0)<0:up=-up
    name=front_image or names[0]
    if name not in names:raise ValueError('Front image must have a registered pose')
    R=np.array(poses['R'][name]);t=np.array(poses['t'][name]);center=np.asarray(mesh.get_axis_aligned_bounding_box().get_center());forward=-R.T@t-center;forward-=up*(forward@up)
    if np.linalg.norm(forward)<1e-8:raise ValueError('Cannot infer horizontal display direction from camera')
    forward/=np.linalg.norm(forward);B=np.column_stack([forward,np.cross(up,forward),up]);T=np.eye(4);T[:3,:3]=B.T;T[:3,3]=-B.T@center;mesh.transform(T);mesh.compute_vertex_normals();return T
