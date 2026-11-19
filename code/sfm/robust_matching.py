"""Batched normalized eight-point verification with ranked and uniform sampling.

The estimator remains our implementation. OpenCV supplies descriptor search only.
This optional quality path is separate from the recorded benchmark configuration.
"""
import math
import numpy as np
from .geometry import fundamental_8pt,sampson_distance


def hypotheses(a,b):
    """Rank-two F matrices for batches of eight paired pixel coordinates."""
    def normalize(x):
        center=x.mean(1);scale=np.sqrt(2)/np.maximum(np.linalg.norm(x-center[:,None],axis=2).mean(1),1e-12)
        p=(x-center[:,None])*scale[:,None,None]
        T=np.zeros((len(x),3,3));T[:,0,0]=T[:,1,1]=scale;T[:,2,2]=1;T[:,:2,2]=-center*scale[:,None]
        return p,T
    x,T1=normalize(a);y,T2=normalize(b)
    u,v=x[:,:,0],x[:,:,1];s,t=y[:,:,0],y[:,:,1]
    A=np.stack([s*u,s*v,s,t*u,t*v,t,u,v,np.ones_like(u)],axis=2)
    _,singular,V=np.linalg.svd(A,full_matrices=True)
    F=V[:,-1].reshape(-1,3,3);U,d,V=np.linalg.svd(F);d[:,-1]=0
    F=(U*d[:,None])@V;F=np.swapaxes(T2,1,2)@F@T1
    F/=np.maximum(np.linalg.norm(F,axis=(1,2)),1e-12)[:,None,None]
    return F,singular[:,-1]>1e-10


def verify_ranked(points1,points2,threshold=1.5,iterations=4000,confidence=.999,seed=0):
    """Deterministic sampled hypotheses, local refits, and conservative stopping."""
    a,b=np.asarray(points1,float),np.asarray(points2,float);n=len(a)
    if n<8:raise ValueError('At least eight matches required')
    rng=np.random.default_rng(seed);x=np.c_[a,np.ones(n)];y=np.c_[b,np.ones(n)]
    best=np.zeros(n,bool);bestF=None;cap=iterations;iteration=0;threshold2=threshold**2
    while iteration<cap:
        count=min(32,cap-iteration);samples=[]
        for k in range(count):
            step=iteration+k
            # Half of trials always sample all matches. The other half starts
            # with stronger descriptor ratios and gradually expands the pool.
            pool=n if step%2 else min(n,16+step//8)
            samples.append(rng.choice(pool,8,replace=False))
        ids=np.asarray(samples);Fs,valid=hypotheses(a[ids],b[ids])
        fx=np.einsum('bij,nj->bni',Fs,x);fty=np.einsum('bji,nj->bni',Fs,y)
        error=(np.einsum('ni,bni->bn',y,fx)**2)/np.maximum((fx[:,:,:2]**2).sum(2)+(fty[:,:,:2]**2).sum(2),1e-12)
        masks=error<=threshold2;scores=masks.sum(1);scores[~valid]=0
        pick=int(np.argmax(scores))
        if scores[pick]>best.sum():
            mask=masks[pick];F=Fs[pick]
            for _ in range(3):
                if mask.sum()<8:break
                fitted=fundamental_8pt(a[mask],b[mask]);refined=sampson_distance(fitted,a,b)<=threshold2
                if refined.sum()<mask.sum():break
                F,mask=fitted,refined
            if mask.sum()>best.sum():
                bestF,best=F,mask
                probability=min(1-1e-12,float(best.mean())**8)
                # Count only the uniform half toward the confidence bound.
                required=2*math.ceil(math.log1p(-confidence)/math.log1p(-probability))
                cap=min(cap,max(iteration+count,128,required))
        iteration+=count
    if bestF is None or best.sum()<8:raise RuntimeError('No valid eight-point consensus')
    return bestF,best
