"""Load the project's common JSON/PLY exports without altering SfM files."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from runtime import write_json

@dataclass
class SceneData:
    image_paths: list[Path]
    K: list[np.ndarray]
    R: list[np.ndarray]
    t: list[np.ndarray]
    registered: list[bool]
    points: np.ndarray
    colors: np.ndarray | None
    tracks: dict | None
    root: Path = Path('.')
    names: list[str] = field(default_factory=list)
    distortion: list[np.ndarray] = field(default_factory=list)
    mask_directory: Path | None = None
    depth_cache: Path | None = None
    inline_images: np.ndarray | None = None
    vggt_path: Path | None = None

    @property
    def extent(self):
        lo,hi=np.percentile(self.points,[2,98],axis=0)
        value=float(np.linalg.norm(hi-lo))
        if not np.isfinite(value) or value<=0: raise ValueError('Scene has no finite spatial extent')
        return value

    def image(self,i):
        if self.inline_images is not None:
            return np.clip(self.inline_images[i]*255,0,255).astype('u1')
        a=np.asarray(Image.open(self.image_paths[i]).convert('RGBA'))
        alpha=a[:,:,3:4]/255.
        rgb=np.rint(a[:,:,:3]*alpha+255*(1-alpha)).astype('u1')
        return cv2.undistort(rgb,self.K[i],self.distortion[i]) if np.any(self.distortion[i]) else rgb

    def mask(self,i):
        if self.mask_directory:
            p=self.mask_directory/(self.names[i]+'.png')
            if not p.exists(): p=self.mask_directory/(Path(self.names[i]).stem+'.png')
            m=cv2.imread(str(p),0)
            if m is None: raise ValueError(f'Missing mask: {p}')
        elif self.inline_images is None:
            m=np.asarray(Image.open(self.image_paths[i]).convert('RGBA'))[:,:,3]
        else: m=np.full(self.inline_images.shape[1:3],255,'u1')
        if m.shape!=self.image(i).shape[:2]:raise ValueError('Mask and image size differ')
        if np.any(self.distortion[i]):m=cv2.undistort(m,self.K[i],self.distortion[i])
        return m>128


def load_scene(path,images=None):
    path=Path(path).resolve()
    if path.suffix=='.npz': return load_vggt(path)
    root=path.parent if path.is_file() else path
    manifest=root/'scene.json'
    meta=json.loads(manifest.read_text()) if manifest.exists() else {}
    recon=(root/meta.get('reconstruction','.')).resolve()
    if not (recon/'poses.json').exists() and (root/'reconstruction/poses.json').exists():recon=root/'reconstruction'
    poses_path=path if path.name=='poses.json' else recon/'poses.json'
    if not poses_path.exists():raise ValueError('Scene needs poses.json and points.ply, or scene.json pointing to them')
    p=json.loads(poses_path.read_text());recon=poses_path.parent
    image_dir=Path(images).resolve() if images else root/meta.get('images','images')
    if not image_dir.is_dir() and images is None:
        candidates=[root/'input/images',root.parent/'input/images',root.parent/'calibrated_2400/images']
        image_dir=next((c for c in candidates if c.is_dir()),image_dir)
    import open3d as o3d
    cloud=o3d.io.read_point_cloud(str(recon/'points.ply'))
    points=np.asarray(cloud.points).copy();colors=np.rint(np.asarray(cloud.colors)*255).astype('u1') if cloud.has_colors() else None
    if len(points)<4 or not np.isfinite(points).all():raise ValueError('Scene requires at least four finite sparse points')
    names=p['image_names'];registered=[n in p['registered'] for n in names];Ks=[];Rs=[];ts=[];paths=[];dist=[]
    import yaml
    calibration=image_dir.parent/'intrinsics.yaml'
    calibrated=yaml.safe_load(calibration.read_text()).get('images',{}) if calibration.exists() else {}
    summary=json.loads((recon/'summary.json').read_text()) if (recon/'summary.json').exists() else {}
    cameras=summary.get('cameras',[])
    if len(cameras)>1 and any(c['model'] not in ('PINHOLE','SIMPLE_PINHOLE') for c in cameras):raise ValueError('Multiple distorted cameras need explicit per-image distortion in poses.json')
    for n,valid in zip(names,registered):
        if valid and (n not in p['R'] or n not in p['t']):raise ValueError(f'Registered image has no pose: {n}')
        f=image_dir/n
        if not f.exists():raise ValueError(f'Missing SfM input image {f}; pass --images or scene.json')
        if valid and n not in p['K']:raise ValueError(f'Registered image has no intrinsics: {n}')
        rawK=p['K'].get(n,calibrated.get(n,{}).get('K',next(iter(p['K'].values()))))
        K=np.asarray(rawK,float);R=np.asarray(p['R'].get(n,np.eye(3)),float);t=np.asarray(p['t'].get(n,[0,0,0]),float).reshape(3)
        if K.shape!=(3,3) or np.any(np.diag(K)[:2]<=0):raise ValueError(f'Invalid K for {n}')
        if valid and (R.shape!=(3,3) or not np.allclose(R.T@R,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(R),1,atol=1e-4)):raise ValueError(f'Invalid world-to-camera rotation for {n}')
        size=list(Image.open(f).size)
        if n in calibrated and 'image_size' in calibrated[n] and calibrated[n]['image_size']!=size:raise ValueError(f'Calibration image size mismatch: {n}')
        if n in p.get('image_size',{}) and p['image_size'][n]!=size:raise ValueError(f'Intrinsics image size mismatch: {n}')
        D=np.asarray(p.get('distortion',{}).get(n,[0,0,0,0,0]),float)
        if len(cameras)==1 and not np.any(D):
            camera=cameras[0]
            if camera['model']=='SIMPLE_RADIAL':D=np.array([camera['params'][3],0,0,0,0.])
            elif camera['model']=='RADIAL':D=np.array([*camera['params'][3:5],0,0,0.])
            elif camera['model'] not in ('PINHOLE','SIMPLE_PINHOLE'):raise ValueError(f"Unsupported camera distortion: {camera['model']}")
        paths.append(f);Ks.append(K);Rs.append(R);ts.append(t);dist.append(D)
    observations=recon/'observations.json';tracks=None
    if observations.exists():
        obs=json.loads(observations.read_text())
        if len(obs)!=len(points):raise ValueError('Observation rows do not match sparse point rows')
        lookup={n:i for i,n in enumerate(names)};tracks={}
        for row,track in enumerate(obs.values()):
            tracks[row]={}
            for n,uv in track.items():
                if n not in lookup:continue
                i=lookup[n];uv=np.asarray(uv,float)
                if np.any(dist[i]):uv=cv2.undistortPoints(uv.reshape(1,1,2),Ks[i],dist[i],P=Ks[i]).ravel()
                tracks[row][i]=uv
    return SceneData(paths,Ks,Rs,ts,registered,points,colors,tracks,root,names,dist,
                     root/meta['masks'] if meta.get('masks') else None,
                     root/meta.get('depth_cache','cache/depth_mono'))


def load_vggt(path):
    with np.load(path,allow_pickle=False) as d:
        if 'depth' not in d or 'images' not in d:raise ValueError('VGGT mesh input requires depth and preprocessed RGB images')
        names=d['image_names'].astype(str).tolist();ext=d['extrinsic'];K=d['intrinsic'];rgb=d['images'].copy()
        if ext.shape!=(len(names),3,4) or K.shape!=(len(names),3,3):raise ValueError('Expected unbatched VGGT camera arrays')
        points=d['world_points'].reshape(-1,3);finite=np.isfinite(points).all(1);points=points[finite][::max(1,int(finite.sum())//50000)]
        if rgb.shape[-1]!=3:raise ValueError('VGGT images must be N,H,W,3, as exported by our adapter')
    return SceneData([path.parent/n for n in names],list(K),list(ext[:,:,:3]),list(ext[:,:,3]),[True]*len(names),points,None,None,path.parent,names,[np.zeros(5)]*len(names),inline_images=rgb,vggt_path=path)


def project(scene,i,points=None):
    X=scene.points if points is None else np.asarray(points)
    q=X@scene.R[i].T+scene.t[i];uv=q@scene.K[i].T
    return uv[:,:2]/np.maximum(q[:,2:3],1e-12),q[:,2]


def visible_samples(scene,i):
    if scene.tracks is None:
        ids=np.arange(len(scene.points));uv,z=project(scene,i)
    else:
        ids=np.array([k for k,v in scene.tracks.items() if i in v],int)
        uv=np.array([scene.tracks[k][i] for k in ids]).reshape(-1,2);z=(scene.points[ids]@scene.R[i].T+scene.t[i])[:,2]
    h,w=scene.image(i).shape[:2];valid=(z>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return ids[valid],uv[valid],z[valid]


def check_convention(scene,out=None,max_error=2.):
    rows=[];errors=[]
    for i,valid in enumerate(scene.registered):
        if not valid:continue
        ids,uv,z=visible_samples(scene,i);projected,depth=project(scene,i,scene.points[ids]);err=np.linalg.norm(projected-uv,axis=1)
        if scene.tracks is not None:errors.extend(err.tolist())
        alluv,allz=project(scene,i);h,w=scene.image(i).shape[:2]
        inside=(allz>0)&(alluv[:,0]>=0)&(alluv[:,0]<w)&(alluv[:,1]>=0)&(alluv[:,1]<h)
        rows.append(dict(image=scene.names[i],positive_fraction=float((allz>0).mean()),inside_fraction=float(inside.mean()),tracked_points=len(ids)))
        if out and len(rows)==1:
            overlay=scene.image(i).copy()
            for p in alluv[inside][::max(1,int(inside.sum())//1500)]:cv2.circle(overlay,tuple(np.rint(p).astype(int)),1,(255,0,0),-1)
            Path(out).mkdir(parents=True,exist_ok=True);cv2.imwrite(str(Path(out)/'projection_overlay.png'),overlay[:,:,::-1])
    median=float(np.median(errors)) if errors else None
    passed=bool(rows) and (median<max_error if median is not None else np.median([r['positive_fraction'] for r in rows])>.8 and np.median([r['inside_fraction'] for r in rows])>.5)
    result=dict(convention='OpenCV world-to-camera; x_cam=R X+t',median_reprojection_px=median,passed=passed,automatic_no_track_check=not bool(errors),scene_extent=scene.extent,images=rows)
    if out:write_json(Path(out)/'convention.json',result)
    if not passed:raise ValueError('Pose convention/reprojection gate failed; inspect convention.json and overlay')
    return result
