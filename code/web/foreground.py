"""Deterministic object masks from supplied alpha or simple capture backgrounds."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageOps


def object_mask(image):
    rgba=np.asarray(image.convert('RGBA'));rgb=rgba[:,:,:3];alpha=rgba[:,:,3]
    if alpha.min()<255:return alpha.copy(),'supplied_alpha'
    h,w=alpha.shape;small=cv2.resize(rgb,(min(w,640),max(1,round(h*min(w,640)/w))))
    sh,sw=small.shape[:2];border=np.concatenate([small[0],small[-1],small[:,0],small[:,-1]])
    if np.quantile(border.max(1),.9)<25:
        mask=(small.max(2)>18).astype('u1')*255;mode='dark_background'
    else:
        # A colored central object on a neutral tabletop: retain a padded hull,
        # including dark tires/legs, rather than accepting only the color pixels.
        hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
        colored=((hsv[:,:,1]>70)&(hsv[:,:,2]>35)&(hsv[:,:,0]>85)&(hsv[:,:,0]<145)).astype('u1')*255
        colored=cv2.morphologyEx(colored,cv2.MORPH_CLOSE,np.ones((5,5),'u1'))
        contours,_=cv2.findContours(colored,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour);x,y,cw,ch=cv2.boundingRect(contour)
            distance=np.linalg.norm([(x+cw/2-sw/2)/sw,(y+ch/2-sh/2)/sh])
            if .004*sw*sh<area<.6*sw*sh and distance<.35:candidates.append((area/(1+20*distance**2),contour))
        if not candidates:return alpha.copy(),'opaque_no_confident_mask'
        hull=cv2.convexHull(max(candidates,key=lambda x:x[0])[1]);mask=np.zeros((sh,sw),'u1');cv2.drawContours(mask,[hull],-1,255,-1)
        _,_,cw,ch=cv2.boundingRect(hull);pad=max(3,round(max(cw,ch)*.09));mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*pad+1,2*pad+1)));mode='central_colored_object'
    count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('u1'))
    if count>1:
        largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask=np.where(labels==largest,255,0).astype('u1')
    fraction=float((mask>0).mean())
    if not .01<fraction<.95:return alpha.copy(),'opaque_mask_gate_rejected'
    mask=cv2.dilate(mask,np.ones((5,5),'u1'))
    return cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST),mode


def prepare_foreground(job):
    """Prepared matching pixels and masks agree; original uploads stay untouched."""
    import json,yaml
    job=Path(job);folder=job/'input/images';target=job/'input_masks';target.mkdir(exist_ok=True);rows=[]
    metadata=json.loads((job/'job.json').read_text());calibration_path=job/'input/intrinsics.yaml'
    calibration=yaml.safe_load(calibration_path.read_text())
    for source in sorted((job/'uploads').iterdir()):
        if not source.is_file():continue
        name=source.stem+'.jpg'
        with Image.open(source) as raw:
            oriented=ImageOps.exif_transpose(raw).convert('RGBA');mask,mode=object_mask(oriented)
            original_w,original_h=oriented.size
            ys,xs=np.where(mask>127)
            if len(xs) and mode not in ('opaque_no_confident_mask','opaque_mask_gate_rejected'):
                padding=round(.05*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1))
                box=[max(0,int(xs.min())-padding),max(0,int(ys.min())-padding),min(original_w,int(xs.max())+padding+1),min(original_h,int(ys.max())+padding+1)]
            else:box=[0,0,original_w,original_h]
            cropped=oriented.convert('RGB').crop(box);crop_mask=Image.fromarray(mask).crop(box)
        with Image.open(folder/name) as prepared:old_w,old_h=prepared.size
        cropped.thumbnail((metadata['max_side'],metadata['max_side']),Image.Resampling.LANCZOS)
        width,height=cropped.size;rgb=np.asarray(cropped).copy();mask=np.asarray(crop_mask.resize(cropped.size,Image.Resampling.NEAREST))
        rgb[mask<128]=0;Image.fromarray(rgb).save(folder/name,quality=95)
        Image.fromarray(mask).save(target/(name+'.png'));Image.fromarray(mask).save(target/(Path(name).stem+'.png'))
        record=calibration['images'][name];K=np.asarray(record['K'],float)
        K[0]*=original_w/old_w;K[1]*=original_h/old_h
        K[0,2]-=box[0];K[1,2]-=box[1]
        K[0]*=width/(box[2]-box[0]);K[1]*=height/(box[3]-box[1])
        record.update(K=K.tolist(),image_size=[width,height])
        rows.append(dict(image=name,source=mode,fraction=float((mask>127).mean()),crop_original=box,original_size=[original_w,original_h],prepared_size=[width,height]))
    calibration_path.write_text(yaml.safe_dump(calibration))
    (job/'foreground.json').write_text(json.dumps(rows,indent=2)+'\n')
    return target
