"""Calibrated CPU stereo, returned as reference-camera z-depth (SfM units)."""
import cv2,numpy as np
from .adapters import project


def stereo_pair(scene,i,j,long_side=640,num_disparities=128):
    rgb1=scene.image(i);rgb2=scene.image(j);h0,w0=rgb1.shape[:2];scale=min(1.,long_side/max(h0,w0));w,h=round(w0*scale),round(h0*scale)
    K1=scene.K[i].copy();K1[:2]*=scale
    K2=scene.K[j].copy();K2[0]*=w/rgb2.shape[1];K2[1]*=h/rgb2.shape[0]
    a=cv2.resize(rgb1,(w,h),interpolation=cv2.INTER_AREA);b=cv2.resize(rgb2,(w,h),interpolation=cv2.INTER_AREA)
    masks=[cv2.resize(scene.mask(k).astype('u1'),(w,h),interpolation=cv2.INTER_NEAREST) for k in [i,j]]
    R=scene.R[j]@scene.R[i].T;t=scene.t[j]-R@scene.t[i]
    r1,r2,p1,p2,_,_,_=cv2.stereoRectify(K1,None,K2,None,(w,h),R,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    if abs(p2[1,3])>abs(p2[0,3]):
        quarter=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
        r1=quarter@r1;r2=quarter@r2
    q1=scene.points@scene.R[i].T+scene.t[i];q2=scene.points@scene.R[j].T+scene.t[j]
    good=(q1[:,2]>0)&(q2[:,2]>0)
    q1r=q1[good]@r1.T;q2r=q2[good]@r2.T;positive=(q1r[:,2]>0)&(q2r[:,2]>0)
    q1r=q1r[positive];q2r=q2r[positive]
    if len(q1r)<10:raise ValueError('Too few common positive-depth stereo landmarks')
    f=min(K1[0,0],K2[0,0]);principal=np.array([w/2,h/2])-np.median(f*q1r[:,:2]/q1r[:,2:],axis=0)
    Kn=np.array([[f,0,principal[0]],[0,f,principal[1]],[0,0,1.]])
    tx=(r2@t)[0]
    if abs(tx)<1e-9:raise ValueError('Stereo pair has zero baseline')
    p1=np.c_[Kn,np.zeros(3)];p2=np.c_[Kn,[f*tx,0,0]]
    maps=[cv2.initUndistortRectifyMap(K1,None,r1,p1,(w,h),cv2.CV_32FC1),cv2.initUndistortRectifyMap(K2,None,r2,p2,(w,h),cv2.CV_32FC1)]
    ims=[cv2.remap(x,*m,cv2.INTER_LINEAR) for x,m in zip([a,b],maps)];ms=[cv2.remap(x,*m,cv2.INTER_NEAREST) for x,m in zip(masks,maps)]
    disparities=f*q1r[:,0]/q1r[:,2]-f*q2r[:,0]/q2r[:,2]
    low=int(np.floor(np.quantile(disparities,.02)))-12;high=int(np.ceil(np.quantile(disparities,.98)))+12
    num=max(num_disparities,((high-low+15)//16)*16);num=min(num,((w//2)//16)*16)
    if low<=-w+20 or low+num>=w-20:raise ValueError('Rectified disparity range lies outside image')
    def matcher(start):return cv2.StereoSGBM_create(minDisparity=int(start),numDisparities=int(num),blockSize=5,P1=8*3*25,P2=32*3*25,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,disp12MaxDiff=1,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    gray=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in ims];dl=matcher(low).compute(*gray).astype('f4')/16;dr=matcher(-low-num).compute(*gray[::-1]).astype('f4')/16
    yy,xx=np.indices((h,w));xr=np.rint(xx-dl).astype(int);inside=(xr>=0)&(xr<w);xr=np.clip(xr,0,w-1)
    valid=inside&(dl>low)&(ms[0]>0)&(ms[1][yy,xr]>0)&(abs(dl+dr[yy,xr])<1.25)
    Q=np.array([[1,0,0,-principal[0]],[0,1,0,-principal[1]],[0,0,0,f],[0,0,-1/tx,0.]])
    rect=cv2.reprojectImageTo3D(dl,Q);cam=rect@r1;valid&=np.isfinite(cam).all(2)&(cam[:,:,2]>0)
    pts=cam[valid];uv=pts@K1.T;uv=np.rint(uv[:,:2]/uv[:,2:]).astype(int);ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    depth=np.full(h*w,np.inf,'f4');np.minimum.at(depth,uv[ok,1]*w+uv[ok,0],pts[ok,2]);depth=depth.reshape(h,w);depth[~np.isfinite(depth)]=0
    return dict(index=i,color=a,depth=depth,K=K1,R=scene.R[i],t=scene.t[i]),dict(neighbor=scene.names[j],valid_pixels=int((depth>0).sum()),baseline=float(np.linalg.norm(t)),disparities=num,min_disparity=low)


def estimate(scene,i,training,config):
    C=-scene.R[i].T@scene.t[i];z=(scene.points@scene.R[i].T+scene.t[i])[:,2];median=float(np.median(z[z>0]));candidates=[]
    for j in training:
        if i==j:continue
        baseline=np.linalg.norm(-scene.R[j].T@scene.t[j]-C)/max(median,1e-9);angle=np.degrees(np.arccos(np.clip(scene.R[i][2]@scene.R[j][2],-1,1)))
        if .015<baseline<.5 and angle<40:candidates.append((abs(baseline-.1)+angle/200,j))
    if not candidates:raise ValueError('No usable stereo neighbor with nonzero baseline')
    frames=[];reports=[]
    for _,j in sorted(candidates)[:config.get('neighbors',2)]:
        try:
            f,r=stereo_pair(scene,i,j,config.get('input_long_side',640),config.get('num_disparities',128));frames.append(f);reports.append(r)
        except (ValueError,cv2.error) as e:reports.append(dict(neighbor=scene.names[j],skipped=str(e)))
    if not frames:raise ValueError('All stereo neighbors failed rectification/disparity checks')
    stack=np.stack([f['depth'] for f in frames]);valid=stack>0;count=valid.sum(0)
    # Average agreeing depth estimates only, retaining single-neighbor coverage.
    denom=np.maximum(count,1);mean=stack.sum(0)/denom;good=((~valid)|(abs(stack-mean)/np.maximum(mean,1e-9)<.05)).all(0)
    return dict(frames[0],depth=np.where(good,mean,0).astype('f4')),dict(pairs=reports)
