"""Memory-bounded VGGT windows; align only their shared predicted cameras."""
from pathlib import Path
import json,logging,os,subprocess,sys
import numpy as np
from runtime import write_json


def windows(count,size=16,anchors=6):
    if count<2 or not 4<=anchors<size:raise ValueError('Invalid VGGT window configuration')
    # Establish one world frame from views spread across the entire capture.
    overview=np.linspace(0,count-1,min(size,count),dtype=int).tolist()
    result=[overview];anchor_ids=[overview[i] for i in np.linspace(0,len(overview)-1,min(anchors,len(overview)),dtype=int)]
    remaining=[i for i in range(count) if i not in set(overview)]
    for start in range(0,len(remaining),size-anchors):result.append(sorted(anchor_ids+remaining[start:start+size-anchors]))
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


def preprocess_masks(paths,height,width=518,mode='crop'):
    """Match official crop/pad geometry, with invalid padding for masks."""
    from PIL import Image
    if mode not in ('crop','pad'):raise ValueError('Unknown VGGT preprocessing mode')
    output=[]
    for path in paths:
        with Image.open(path) as image:
            if mode=='pad' and image.height>image.width:h=518;w=round(image.width*518/image.height/14)*14
            else:w=518;h=round(image.height*518/image.width/14)*14
            mask=np.array(image.convert('L').resize((w,h),Image.Resampling.BILINEAR))
        if mode=='crop' and h>518:mask=mask[(h-518)//2:(h-518)//2+518];h=518
        top=(height-h)//2;left=(width-w)//2
        if top<0 or left<0:raise ValueError('Mask preprocessing shape mismatch')
        output.append(np.pad(mask,((top,height-h-top),(left,width-w-left)))>127)
    return np.stack(output)


def validate_cached_inputs(folder,names,paths,preprocess_mode='crop'):
    import hashlib
    metadata=json.loads((Path(folder)/'raw.json').read_text())
    if metadata.get('preprocessing_mode','crop')!=preprocess_mode:raise ValueError('Cached VGGT preprocessing differs; run fresh inference')
    expected={p.name:p for p in paths}
    if len(names)>16 or len(names)!=len(set(names)) or any(n not in expected for n in names):raise ValueError('Cached window names/context do not match this input')
    for name in names:
        original=Path(folder)/'images'/name
        if not original.is_file() or hashlib.sha256(original.read_bytes()).digest()!=hashlib.sha256(expected[name].read_bytes()).digest():
            raise ValueError('Cached VGGT window pixels differ from current prepared inputs')


def infer_batched(images,out,cached_windows=None,preprocess_mode='crop'):
    images,out=Path(images),Path(out);paths=sorted(p for p in images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    plan=windows(len(paths));root=out.parent/'windows';root.mkdir(parents=True,exist_ok=True);stages=[];selected={};parts=[]
    overview_names={paths[i].name for i in plan[0]}
    if cached_windows:
        cached=sorted(Path(cached_windows).glob('*/raw.npz'))
        if not cached:raise ValueError('No saved VGGT windows are available to realign')
        work=[(plan[0],None)]+[(None,p.parent) for p in cached]
    else:work=[(indices,None) for indices in plan]
    for wi,(indices,cache) in enumerate(work):
        folder=root/f'{wi:02d}';inputs=folder/'images';inputs.mkdir(parents=True,exist_ok=True)
        if cache:
            with np.load(cache/'raw.npz',allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
            names=arrays['image_names'].astype(str).tolist();validate_cached_inputs(cache,names,paths,preprocess_mode)
            indices=[next(i for i,p in enumerate(paths) if p.name==name) for name in names]
            for name in names:os.link(cache/'images'/name,inputs/name)
            for suffix in ('npz','json'):os.link(cache/f'raw.{suffix}',folder/f'raw.{suffix}')
        else:
            for index in indices:os.link(paths[index],inputs/paths[index].name)
        logging.info('VGGT window %d/%d: %d views (total input %d)%s',wi+1,len(work),len(indices),len(paths),' — aligning cached predictions' if cache else '')
        if not cache:
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'baselines/vggt_infer.py'),'--images',str(inputs),'--out',str(folder/'raw.npz'),'--preprocess-mode',preprocess_mode],check=True)
            with np.load(folder/'raw.npz',allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
        names=arrays['image_names'].astype(str).tolist();alignment=None
        if wi:
            # Every local model aligns directly to the globally spread overview,
            # rather than accumulating transforms along a chain of local windows.
            shared=[(i,selected[name]) for i,name in enumerate(names) if name in overview_names]
            local=arrays['extrinsic'][[i for i,_ in shared]];reference=np.asarray([parts[pi]['extrinsic'][ii] for _,(pi,ii) in shared])
            s,Q,shift,alignment=camera_alignment(local,reference);arrays=transform(arrays,s,Q,shift)
        keep=[i for i,name in enumerate(names) if name not in selected];slot=len(parts)
        trimmed={key:value[keep] for key,value in arrays.items()};parts.append(trimmed)
        for ii,name in enumerate(trimmed['image_names'].astype(str)):selected[name]=(slot,ii)
        stages.append(dict(indices=indices,alignment=alignment,prediction_reused=bool(cache),metadata=json.loads((folder/'raw.json').read_text())))
        write_json(root/'alignment.json',dict(global_overview=sorted(overview_names),windows=stages))
    if len(selected)!=len(paths):raise ValueError('Batched VGGT omitted an input view')
    height=max(part['images'].shape[1] for part in parts);parts=[pad_predictions(part,height) for part in parts]
    names=[p.name for p in paths];merged={}
    for key in parts[0]:merged[key]=np.stack([parts[selected[name][0]][key][selected[name][1]] for name in names])
    np.savez_compressed(out,**merged)
    meta=dict(mode='Global 16-view overview plus overlapping local VGGT windows; direct shared predicted-camera Sim(3) alignment',
              joint_all_views=False,max_joint_views=16,preprocessing_mode=preprocess_mode,total_images=len(paths),windows=stages,partially_reused_predictions=bool(cached_windows),
              warning='Window stitching can introduce seams; this is not a single joint 60-view inference. Global camera alignment is checked before fusion.')
    write_json(out.with_suffix('.json'),meta);return meta
