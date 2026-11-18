"""Optional shared-focal BA; default fixed-intrinsics BA remains unchanged."""
import numpy as np


def bundle_adjust_focal(rotations,translations,points,camera_indices,point_indices,observations,intrinsics,
                        fixed_camera_ids=(0,),scale_camera_ids=(0,1),loss='huber',f_scale=1.,max_nfev=100):
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from .bundle_adjust import BAResult,_rvecs_from_rotations,_rotations_from_rvecs,_project
    R0=np.asarray(rotations,float).copy();t0=np.asarray(translations,float).reshape(-1,3).copy()
    X0=np.asarray(points,float).copy();ci=np.asarray(camera_indices,int);pi=np.asarray(point_indices,int)
    obs=np.asarray(observations,float);K=np.asarray(intrinsics,float)
    a,b=scale_camera_ids
    def baseline(R,t):
        centers=-np.einsum('nji,nj->ni',R[[a,b]],t[[a,b]])
        return np.linalg.norm(centers[0]-centers[1])
    baseline0=baseline(R0,t0)
    if baseline0<=1e-12:raise ValueError('Scale anchors must have nonzero baseline')
    weight=float(np.median(K[:,0,0]))/baseline0
    free=np.array([i for i in range(len(R0)) if i not in fixed_camera_ids],int)
    slots={camera:slot for slot,camera in enumerate(free)};rvec0=_rvecs_from_rotations(R0)
    n=len(free);point_end=6*n+3*len(X0)
    def unpack(x):
        r,t=rvec0.copy(),t0.copy();r[free]=x[:3*n].reshape(n,3);t[free]=x[3*n:6*n].reshape(n,3)
        return _rotations_from_rvecs(r),t,x[6*n:point_end].reshape(-1,3),float(np.exp(x[-1]))
    def residual(x):
        R,t,X,scale=unpack(x);fitted=K.copy();fitted[:,0,0]*=scale;fitted[:,1,1]*=scale
        pixels=(_project(X[pi],R[ci],t[ci],fitted)-obs).ravel()
        return np.r_[pixels,weight*(baseline(R,t)-baseline0)]
    sparsity=lil_matrix((2*len(obs)+1,point_end+1),dtype=int);rows=np.arange(len(obs))
    for camera,slot in slots.items():
        matched=rows[ci==camera]
        for column in range(6):
            offset=3*slot+column if column<3 else 3*n+3*slot+column-3
            sparsity[2*matched,offset]=1;sparsity[2*matched+1,offset]=1
    for column in range(3):
        sparsity[2*rows,6*n+3*pi+column]=1;sparsity[2*rows+1,6*n+3*pi+column]=1
    sparsity[:2*len(obs),-1]=1
    for camera in (a,b):
        if camera in slots:
            slot=slots[camera];sparsity[-1,3*slot:3*slot+3]=1;sparsity[-1,3*n+3*slot:3*n+3*slot+3]=1
    x0=np.r_[rvec0[free].ravel(),t0[free].ravel(),X0.ravel(),0.]
    lower=np.full(len(x0),-np.inf);upper=np.full(len(x0),np.inf)
    lower[-1]=np.log(.5);upper[-1]=np.log(2.)
    initial=residual(x0)[:-1]
    # Initialize the single focal multiplier from current camera/point
    # projections. This uses only observations, never reference calibration.
    # Starting with a gross focal error can otherwise stall robust sparse BA.
    principal=K[:,:2,2]
    predicted=_project(X0[pi],R0[ci],t0[ci],K)-principal
    target=obs-principal
    finite=np.isfinite(predicted).all(1)&np.isfinite(target).all(1)
    p=predicted[finite].ravel();y=target[finite].ravel()
    multiplier=1.
    for _ in range(5):
        weights=np.minimum(1.,max(f_scale,1e-6)/np.maximum(abs(multiplier*p-y),1e-8))
        denominator=float((weights*p*p).sum())
        if denominator>1e-12:multiplier=float((weights*p*y).sum()/denominator)
    x0[-1]=np.log(np.clip(multiplier,.500001,1.999999))
    fit=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),method='trf',tr_solver='lsmr',x_scale='jac',
                      loss=loss,f_scale=f_scale,max_nfev=max_nfev,bounds=(lower,upper),
                      tr_options=dict(atol=1e-10,btol=1e-10,maxiter=2000))
    R,t,X,scale=unpack(fit.x);final=residual(fit.x)[:-1]
    return BAResult(R,t,X,float(np.sqrt(np.mean(initial**2))),float(np.sqrt(np.mean(final**2))),fit.nfev,scale)
