"""Robust inverse-depth scale/shift alignment to visible SfM landmarks."""
import cv2
import numpy as np
from .adapters import visible_samples


class AlignmentError(ValueError):
    def __init__(self,message,details):
        super().__init__(message);self.details=details

def aligned_depth(relative,s,b):
    inv=s*np.asarray(relative,dtype=float)+b
    return np.divide(1.,inv,out=np.zeros_like(inv),where=np.isfinite(inv)&(inv>1e-6))


def fit_inverse_depth_affine(d_rel,z_sfm,iters=200,tol=.05,seed=0):
    x=np.asarray(d_rel,float).ravel();z=np.asarray(z_sfm,float).ravel()
    valid=np.isfinite(x)&np.isfinite(z)&(z>0)
    if valid.sum()<2 or np.ptp(x[valid])<1e-8:raise ValueError('Too few varying finite depth samples')
    ids=np.flatnonzero(valid);y=1/z;best=np.zeros(len(z),bool);rng=np.random.default_rng(seed)
    for _ in range(iters):
        i,j=rng.choice(ids,2,replace=False)
        if abs(x[i]-x[j])<1e-9:continue
        s=(y[i]-y[j])/(x[i]-x[j]);b=y[i]-s*x[i];pred=aligned_depth(x,s,b)
        inliers=valid&(pred>0)&(abs(pred-z)/np.maximum(z,1e-12)<tol)
        if inliers.sum()>best.sum():best=inliers
    if best.sum()<2:raise ValueError('Inverse-depth alignment has no robust consensus')
    for _ in range(3):
        s,b=np.linalg.lstsq(np.column_stack([x[best],np.ones(best.sum())]),y[best],rcond=None)[0]
        pred=aligned_depth(x,s,b);new=valid&(pred>0)&(abs(pred-z)/np.maximum(z,1e-12)<tol)
        if new.sum()<2:break
        best=new
    residual=float(np.median(abs(aligned_depth(x[best],s,b)-z[best])/z[best]))
    return float(s),float(b),best,residual


def align_image(scene,i,relative,config,seed=0):
    ids,uv,z=visible_samples(scene,i)
    h,w=scene.image(i).shape[:2];dh,dw=relative.shape
    xy=uv*np.array([dw/w,dh/h]);sample=cv2.remap(relative.astype('f4'),xy[:,0].astype('f4').reshape(-1,1),xy[:,1].astype('f4').reshape(-1,1),cv2.INTER_LINEAR).ravel()
    if len(ids)<config['min_points']:raise ValueError(f'Only {len(ids)} visible landmarks for alignment')
    s,b,inliers,res=fit_inverse_depth_affine(sample,z,iters=config.get('iterations',400),tol=config.get('ransac_tol',.10),seed=seed)
    details=dict(s=s,b=b,n_inliers=int(inliers.sum()),samples=len(ids),median_rel_residual=res)
    if inliers.sum()<config['min_points'] or res>config['max_median_rel_residual']:raise AlignmentError(f'Alignment quality gate: {inliers.sum()} inliers, median relative error {res:.3g}',details)
    return aligned_depth(relative,s,b).astype('f4'),details
