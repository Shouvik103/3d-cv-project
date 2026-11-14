"""Shared depth cleanup, sparse-object ROI and visibility-aware consistency."""
import cv2,numpy as np


def roi_bounds(points,percentiles=(5,95),margin=.15):
    lo,hi=np.percentile(points,percentiles,axis=0);pad=(hi-lo)*margin
    return lo-pad,hi+pad


def backproject(depth,K,R,t):
    yy,xx=np.indices(depth.shape);rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
    return (rays*depth[:,:,None]-t)@R


def filter_depth(frame,roi,config,mask=None):
    d=frame['depth'].copy();valid=np.isfinite(d)&(d>0)
    if mask is not None:valid&=cv2.resize(mask.astype('u1'),(d.shape[1],d.shape[0]),interpolation=cv2.INTER_NEAREST)>0
    if not valid.any():return np.zeros_like(d)
    trunc=float(np.percentile(d[valid],config.get('depth_trunc_pct',98)))
    valid&=d<=trunc
    # Compare adjacent valid depths; invalid/background pixels never contaminate gradients.
    gradient=np.zeros_like(d)
    for axis in [0,1]:
        delta=abs(np.diff(d,axis=axis));a=[slice(None)]*2;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1)
        rel=delta/np.maximum(np.minimum(d[tuple(a)],d[tuple(b)]),1e-8);rel[(d[tuple(a)]<=0)|(d[tuple(b)]<=0)]=0
        gradient[tuple(a)]=np.maximum(gradient[tuple(a)],rel);gradient[tuple(b)]=np.maximum(gradient[tuple(b)],rel)
    valid&=gradient<=config.get('edge_grad_thresh',.05)
    X=backproject(d,frame['K'],frame['R'],frame['t']);valid&=((X>=roi[0])&(X<=roi[1])).all(2)
    return np.where(valid,d,0).astype('f4')


def consistency_filter(frames,rel_tol=.02,min_views=2,neighbors=4):
    output=[]
    for i,f in enumerate(frames):
        d=f['depth'];X=backproject(d,f['K'],f['R'],f['t']).reshape(-1,3);support=np.zeros(d.size,'u1');seen=np.zeros(d.size,'u1')
        # Angular neighbors with distinct camera centers; compare frozen input depths.
        order=sorted((j for j in range(len(frames)) if j!=i),key=lambda j:-float(f['R'][2]@frames[j]['R'][2]))[:neighbors]
        for j in order:
            g=frames[j];q=X@g['R'].T+g['t'];uv=q@g['K'].T;uv=uv[:,:2]/np.maximum(q[:,2:],1e-12)
            sampled=cv2.remap(g['depth'],uv[:,0].astype('f4').reshape(d.shape),uv[:,1].astype('f4').reshape(d.shape),cv2.INTER_NEAREST).ravel()
            valid=(q[:,2]>0)&(sampled>0)
            # Points behind an observed neighbor surface are occluded, not inconsistent.
            visible=valid&(q[:,2]<=sampled*(1+rel_tol))
            seen+=visible;support+=visible&(abs(q[:,2]-sampled)/np.maximum(q[:,2],1e-9)<rel_tol)
        need=np.where(seen<=2,1,min_views)
        keep=(seen>0)&(support>=need)
        output.append(dict(f,depth=np.where(keep.reshape(d.shape),d,0).astype('f4')))
    return output
