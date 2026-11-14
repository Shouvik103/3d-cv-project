"""Mesh geometry statistics, held-out images and aligned geometry comparisons."""
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree


def mesh_statistics(mesh):
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);_,components,_=mesh.cluster_connected_triangles()
    return dict(vertices=len(v),faces=len(f),components=len(components),boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts>2).sum()),watertight=bool(mesh.is_watertight()),surface_area=float(mesh.get_surface_area()),mean_edge_length=float(np.linalg.norm(v[unique[:,0]]-v[unique[:,1]],axis=1).mean()) if len(unique) else 0.)


def sample_surface(mesh,count,seed):
    rng=np.random.default_rng(seed);v=np.asarray(mesh.vertices);t=np.asarray(mesh.triangles);p=v[t];area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    if area.sum()<=0:raise ValueError('Mesh has zero surface area')
    ids=rng.choice(len(t),count,p=area/area.sum());uv=rng.random((count,2));flip=uv.sum(1)>1;uv[flip]=1-uv[flip]
    return p[ids,0]+uv[:,:1]*(p[ids,1]-p[ids,0])+uv[:,1:]*(p[ids,2]-p[ids,0])


def compare_meshes(estimate,reference,samples=200000,seed=0,sim3=None,crop=False):
    est=deepcopy(estimate)
    if sim3 is not None:
        scale,Q,b=sim3;T=np.eye(4);T[:3,:3]=scale*Q;T[:3,3]=b;est.transform(T)
    if crop:est=est.crop(reference.get_axis_aligned_bounding_box())
    a=sample_surface(est,samples,seed);b=sample_surface(reference,samples,seed+1);da=cKDTree(b).query(a)[0];db=cKDTree(a).query(b)[0];extent=float(np.linalg.norm(reference.get_axis_aligned_bounding_box().get_extent()))
    scores={}
    for frac in [.005,.01,.02]:
        precision=float((da<extent*frac).mean());recall=float((db<extent*frac).mean());scores[str(frac)]=dict(threshold=extent*frac,precision=precision,recall=recall,f_score=2*precision*recall/(precision+recall) if precision+recall else 0.)
    return dict(accuracy_mean=float(da.mean()),completeness_mean=float(db.mean()),chamfer_mean=float((da.mean()+db.mean())/2),thresholds=scores,samples=samples,seed=seed)


def heldout_metrics(mesh,scene,indices):
    from skimage.metrics import structural_similarity
    from .rendering import Renderer
    renderer=Renderer(mesh);rows=[]
    for i in indices:
        target=scene.image(i);h,w=target.shape[:2];r=renderer.render(scene.K[i],scene.R[i],scene.t[i],w,h);mask=r['mask'];count=int(mask.sum())
        if count:
            error=(r['color'].astype(float)-target)/255.;mse=float((error[mask]**2).mean());psnr=float(-10*np.log10(max(mse,1e-12)))
            _,ssim=structural_similarity(target/255.,r['color']/255.,channel_axis=2,data_range=1.,full=True)
            # A full SSIM window must hit geometry, so background cannot inflate the score.
            import cv2
            interior=cv2.erode(mask.astype('u1'),np.ones((7,7),'u1'))>0
            score=float(ssim[interior].mean()) if interior.any() else None
            rows.append(dict(image=scene.names[i],pixels=count,coverage=float(mask.mean()),psnr_db=psnr,ssim=score,mae=float(abs(error[mask]).mean())))
        else:rows.append(dict(image=scene.names[i],pixels=0,coverage=0.,psnr_db=None,ssim=None,mae=None))
    return dict(views=len(rows),rows=rows,scope='Views excluded from depth generation/consistency/fusion; SfM poses and sparse landmarks were estimated using all input views')


def camera_alignment(estimate,reference):
    from eval.align import umeyama
    shared=sorted(set(estimate.names[i] for i,v in enumerate(estimate.registered) if v)&set(reference.names[i] for i,v in enumerate(reference.registered) if v))
    def centers(s):
        lookup={n:i for i,n in enumerate(s.names)};return np.array([-s.R[lookup[n]].T@s.t[lookup[n]] for n in shared])
    return umeyama(centers(estimate),centers(reference))


def main():
    from pathlib import Path
    import sys,argparse
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from mesh.adapters import load_scene
    from mesh.mesh_export import read_mesh
    from runtime import write_json
    p=argparse.ArgumentParser(description='Camera-aligned mesh comparison; agreement is not ground truth')
    p.add_argument('--estimate',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--estimate-scene',type=Path);p.add_argument('--reference-scene',type=Path);p.add_argument('--same-frame',action='store_true');p.add_argument('--kind',choices=['gt','agreement'],default='agreement');p.add_argument('--samples',type=int,default=200000);p.add_argument('--seed',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not a.same_frame and (not a.estimate_scene or not a.reference_scene):p.error('Provide both camera scenes for Sim(3), or explicitly declare --same-frame')
    transform=None if a.same_frame else camera_alignment(load_scene(a.estimate_scene),load_scene(a.reference_scene))
    metrics=compare_meshes(read_mesh(a.estimate),read_mesh(a.reference),a.samples,a.seed,transform,crop=a.kind=='gt')
    metrics.update(kind=a.kind,alignment='explicit common frame' if a.same_frame else 'Umeyama on shared camera centers',reference=a.reference.name)
    write_json(a.out,metrics)

if __name__=='__main__':main()
