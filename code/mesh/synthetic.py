"""Analytic fixtures, explicitly separate from measured real-scene evidence."""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image
from sfm.io_utils import save_ply
from runtime import write_json


def look_at(eye,target=np.zeros(3),up=np.array([0.,0.,1.])):
    forward=np.asarray(target)-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
    R=np.stack([right,down,forward]);return R,-R@eye


def analytic_scene(shape='sphere',size=96,views=12,pixel_center=.5):
    K=np.array([[size*1.05,0,size/2],[0,size*1.05,size/2],[0,0,1.]])
    yy,xx=np.indices((size,size));rays=np.stack([xx+pixel_center,yy+pixel_center,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    frames=[]
    for i,a in enumerate(np.linspace(0,2*np.pi,views,endpoint=False)):
        eye=np.array([3*np.cos(a)*np.cos(.3),3*np.sin(a)*np.cos(.3),3*np.sin(.3)])
        R,t=look_at(eye);d=rays@R
        if shape=='sphere':
            A=(d*d).sum(2);B=2*(d*eye).sum(2);C=eye@eye-1;disc=B*B-4*A*C
            depth=np.where(disc>=0,(-B-np.sqrt(np.maximum(disc,0)))/(2*A),0)
        else:
            safe=np.where(abs(d)>1e-12,d,1e-12);a0=(-.75-eye)/safe;a1=(.75-eye)/safe;near=np.minimum(a0,a1).max(2);far=np.maximum(a0,a1).min(2);depth=np.where((far>=near)&(near>0),near,0)
        X=rays*depth[:,:,None]@R+eye
        color=np.clip((X+1)/2*255,0,255).astype('u1');color[depth<=0]=245
        frames.append(dict(index=i,color=color,depth=depth.astype('f4'),K=K.copy(),R=R,t=t))
    return frames


def write_sample(out,size=96,views=12,shape='sphere'):
    out=Path(out);(out/'images').mkdir(parents=True,exist_ok=True);(out/'cache/depth_mono').mkdir(parents=True,exist_ok=True)
    frames=analytic_scene(shape,size,views,pixel_center=0);rng=np.random.default_rng(0)
    X=rng.normal(size=(2500,3));X/=np.linalg.norm(X,axis=1,keepdims=True)
    if shape=='box':X*=.75/np.abs(X).max(1,keepdims=True)
    names=[f'view_{i:02}.png' for i in range(views)];p=dict(image_names=names,registered=names,K={},R={},t={},image_size={});obs={str(k):{} for k in range(len(X))}
    for n,f in zip(names,frames):
        Image.fromarray(f['color']).save(out/'images'/n)
        for key in ['K','R','t']:p[key][n]=f[key].tolist()
        p['image_size'][n]=[size,size]
        q=X@f['R'].T+f['t'];uv=(q@f['K'].T)[:,:2]/q[:,2:]
        C=-f['R'].T@f['t'];visible=(X*(C-X)).sum(1)/np.linalg.norm(C-X,axis=1)>.35
        for k in np.flatnonzero(visible):obs[str(k)][n]=uv[k].tolist()
    save_ply(X,np.clip((X+1)*127.5,0,255).astype('u1'),out/'points.ply');write_json(out/'poses.json',p);write_json(out/'observations.json',obs)
    # Exact analytic relative inverse depth with a known affine transform.
    from .depth_mono import cache_key,save_cache
    for n,f in zip(names,frames):
        relative=np.zeros_like(f['depth']);good=f['depth']>0;relative[good]=(1/f['depth'][good]-.1)/1.7
        cfg=dict(model_id='analytic-fixture',input_long_side=size)
        key=cache_key(out/'images'/n,cfg);save_cache(out/'cache/depth_mono'/f'{key}.npz',relative,dict(key=key,**cfg,provenance='Analytic ray-surface fixture; not a learned prediction'))
    write_json(out/'scene.json',dict(images='images',depth_cache='cache/depth_mono',synthetic=True,shape=shape))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--shape',choices=['sphere','box'],default='sphere');a=p.parse_args();write_sample(a.out,shape=a.shape)
