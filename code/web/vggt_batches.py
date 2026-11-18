"""Memory-bounded VGGT windows; align only their shared predicted cameras."""
from pathlib import Path
import json,logging,os,subprocess,sys
import numpy as np
from runtime import write_json


def windows(count,size=16,overlap=5,anchors=3):
    if count<2 or size<=overlap+anchors:raise ValueError('Invalid VGGT window configuration')
    result=[list(range(min(size,count)))];start=size
    anchor_ids=np.linspace(0,min(size,count)-1,anchors,dtype=int).tolist()
    while start<count:
        ids=sorted(set(anchor_ids+list(range(max(0,start-overlap),min(count,start+size-overlap-anchors)))))
        result.append(ids);start=max(ids)+1
    return result


def centers(ext):return -np.einsum('nji,nj->ni',ext[:,:,:3],ext[:,:,3])


def camera_alignment(local,reference):
    """Fit local->reference Sim(3); reject degenerate or inconsistent overlaps."""
    if len(local)<4 or local.shape!=reference.shape:raise ValueError('Need four shared cameras')
    a,b=centers(local),centers(reference);ac,bc=a-a.mean(0),b-b.mean(0)
    u,d,vt=np.linalg.svd(bc.T@ac/len(a));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    if d[1]<1e-10 or np.mean(ac**2)<1e-10:raise ValueError('Degenerate shared-camera layout')
    Q=u@np.diag(sign)@vt;s=float((d*sign).sum()/np.mean(np.sum(ac**2,axis=1)));shift=b.mean(0)-s*Q@a.mean(0)
    aligned=s*a@Q.T+shift;extent=max(np.linalg.norm(np.ptp(b,axis=0)),1e-9)
    center_error=float(np.sqrt(np.mean(np.sum((aligned-b)**2,axis=1)))/extent)
    rotations=local[:,:,:3]@Q.T
    angle=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',rotations,reference[:,:,:3])-1)/2,-1,1)))
    rotation_error=float(np.median(angle))
    if not np.isfinite([s,center_error,rotation_error]).all() or s<=0 or center_error>.25 or rotation_error>30:
        raise ValueError(f'VGGT window alignment failed: normalized camera RMS {center_error:.3f}, median rotation {rotation_error:.1f} degrees')
    return s,Q,shift,dict(shared_cameras=len(a),scale=s,normalized_center_rms=center_error,median_rotation_degrees=rotation_error)


def transform(arrays,s,Q,shift):
    result=dict(arrays);ext=arrays['extrinsic'].copy();ext[:,:,:3]=ext[:,:,:3]@Q.T
    ext[:,:,3]=s*arrays['extrinsic'][:,:,3]-np.einsum('nij,j->ni',ext[:,:,:3],shift)
    result['extrinsic']=ext;result['depth']=arrays['depth']*s
    result['world_points']=(s*arrays['world_points']@Q.T+shift).astype(np.float32)
    return result


def pad_predictions(arrays,height):
    """Official crop batches can have different padded heights; preserve K."""
    h=arrays['images'].shape[1]
    if h==height:return arrays
    if h>height:raise ValueError('Cannot pad predictions to a smaller height')
    top=(height-h)//2;bottom=height-h-top;result=dict(arrays)
    for key in ('images','world_points','world_points_conf','depth','depth_conf'):
        value=arrays[key];padding=[(0,0),(top,bottom),(0,0)]+[(0,0)]*(value.ndim-3)
        result[key]=np.pad(value,padding,constant_values=1 if key=='images' else 0)
    result['intrinsic']=arrays['intrinsic'].copy();result['intrinsic'][:,1,2]+=top
    return result


def preprocess_masks(paths,height,width=518):
    """Match official crop geometry, with invalid (zero) padding for masks."""
    from PIL import Image
    output=[]
    for path in paths:
        with Image.open(path) as image:
            h=round(image.height*518/image.width/14)*14
            mask=np.array(image.convert('L').resize((518,h),Image.Resampling.BILINEAR))
        if h>518:mask=mask[(h-518)//2:(h-518)//2+518];h=518
        top=(height-h)//2
        if top<0 or width!=518:raise ValueError('Mask preprocessing shape mismatch')
        output.append(np.pad(mask,((top,height-h-top),(0,0)))>127)
    return np.stack(output)


def infer_batched(images,out):
    images,out=Path(images),Path(out);paths=sorted(p for p in images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    plan=windows(len(paths));root=out.parent/'windows';root.mkdir(parents=True,exist_ok=True);stages=[];selected={};parts=[]
    for wi,indices in enumerate(plan):
        folder=root/f'{wi:02d}';inputs=folder/'images';inputs.mkdir(parents=True,exist_ok=True)
        for index in indices:
            target=inputs/paths[index].name
            if not target.exists():os.link(paths[index],target)
        logging.info('VGGT window %d/%d: %d views (total input %d)',wi+1,len(plan),len(indices),len(paths))
        subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'baselines/vggt_infer.py'),'--images',str(inputs),'--out',str(folder/'raw.npz')],check=True)
        with np.load(folder/'raw.npz',allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
        names=arrays['image_names'].astype(str).tolist();alignment=None
        if wi:
            shared=[(i,selected[name]) for i,name in enumerate(names) if name in selected]
            local=arrays['extrinsic'][[i for i,_ in shared]];reference=np.asarray([parts[pi]['extrinsic'][ii] for _,(pi,ii) in shared])
            s,Q,shift,alignment=camera_alignment(local,reference);arrays=transform(arrays,s,Q,shift)
        keep=[i for i,name in enumerate(names) if name not in selected]
        trimmed={key:value[keep] for key,value in arrays.items()};parts.append(trimmed)
        for ii,name in enumerate(trimmed['image_names'].astype(str)):selected[name]=(wi,ii)
        stages.append(dict(indices=indices,alignment=alignment,metadata=json.loads((folder/'raw.json').read_text())))
    height=max(part['images'].shape[1] for part in parts)
    parts=[pad_predictions(part,height) for part in parts]
    names=[p.name for p in paths];merged={}
    for key in parts[0]:merged[key]=np.stack([parts[selected[name][0]][key][selected[name][1]] for name in names])
    if len(selected)!=len(paths):raise ValueError('Batched VGGT omitted an input view')
    np.savez_compressed(out,**merged)
    meta=dict(mode='Overlapping 16-view VGGT windows with shared predicted-camera Sim(3) alignment',
              joint_all_views=False,max_joint_views=16,total_images=len(paths),windows=stages,
              warning='Window stitching can introduce seams; this is not a single joint 60-view inference. Shared-camera alignment is checked before fusion.')
    write_json(out.with_suffix('.json'),meta);return meta
