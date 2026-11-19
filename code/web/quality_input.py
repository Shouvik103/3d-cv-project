"""Full-sensor classical inputs and explicit lens priors for quality-first runs.

All pixels come from this job's uploads. A prior contains intrinsics only, never
camera poses or scene geometry. Portrait rotation is recorded and repeated for
mesh photos. Classical quality inputs retain the full camera frame.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageOps
import yaml
from prepare_data import estimate_intrinsics
from web.foreground import object_mask


def prepare_quality_input(job,long_side,target=None,record=True):
    job=Path(job);target=Path(target) if target else job/'input';images=target/'images';images.mkdir(parents=True,exist_ok=True)
    masks=target/'masks';masks.mkdir(exist_ok=True);features=target/'feature_masks';features.mkdir(exist_ok=True)
    metadata=json.loads((job/'job.json').read_text());prior=metadata.get('camera_prior');records={};rows=[]
    for source in sorted((job/'uploads').iterdir()):
        if not source.is_file():continue
        with Image.open(source) as raw:
            oriented=ImageOps.exif_transpose(raw);original_size=oriented.size;rgba=np.array(oriented.convert('RGBA'))
            if prior:
                f=prior['focal_ratio']*max(original_size);K=np.array([[f,0,original_size[0]/2],[0,f,original_size[1]/2],[0,0,1.]])
                D=np.asarray(prior.get('distortion',[0,0,0,0,0]),float);calibration_source='intrinsic_prior_only: '+prior['source']
            elif metadata.get('camera_matrix'):
                K=np.asarray(metadata['camera_matrix'],float).copy();D=np.zeros(5);calibration_source='user_camera_matrix_at_original_resolution'
            elif metadata.get('focal_px'):
                f=metadata['focal_px'];K=np.array([[f,0,original_size[0]/2],[0,f,original_size[1]/2],[0,0,1.]])
                D=np.zeros(5);calibration_source='user_focal_px_at_original_resolution'
            else:K,calibration_source=estimate_intrinsics(oriented);D=np.zeros(5)
        # Alpha compositing matches the successful saved chair's input policy.
        alpha=rgba[:,:,3].copy();supplied_alpha=bool(alpha.min()<255);rgb=np.rint(rgba[:,:,:3]*(alpha[:,:,None]/255)+255*(1-alpha[:,:,None]/255)).astype('u1')
        if np.any(D):
            rgb=cv2.undistort(rgb,K,D);alpha=cv2.undistort(alpha,K,D)
        rotation=90 if rgb.shape[0]>rgb.shape[1] else 0
        if rotation:
            rgb=cv2.rotate(rgb,cv2.ROTATE_90_CLOCKWISE);alpha=cv2.rotate(alpha,cv2.ROTATE_90_CLOCKWISE)
            old=K.copy();K=np.array([[old[1,1],0,rgb.shape[1]/2],[0,old[0,0],rgb.shape[0]/2],[0,0,1.]])
            if metadata.get('camera_matrix'):
                K[0,2]=original_size[1]-1-old[1,2];K[1,2]=old[0,2]
        transformed_size=[rgb.shape[1],rgb.shape[0]]
        mask,mode=object_mask(Image.fromarray(np.dstack([rgb,alpha])) if supplied_alpha else Image.fromarray(rgb))
        mask=np.where(alpha>127,mask,0).astype('u1')
        box=[0,0,*transformed_size]
        rgb=rgb[box[1]:box[3],box[0]:box[2]].copy();mask=mask[box[1]:box[3],box[0]:box[2]]
        K[0,2]-=box[0];K[1,2]-=box[1]
        photo=Image.fromarray(rgb);photo.thumbnail((long_side,long_side),Image.Resampling.LANCZOS)
        width,height=photo.size;K[0]*=width/rgb.shape[1];K[1]*=height/rgb.shape[0]
        rgb=np.array(photo);mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST);rgb[mask<128]=255
        name=source.stem+'.jpg';Image.fromarray(rgb).save(images/name,quality=95)
        Image.fromarray(mask).save(masks/(name+'.png'));Image.fromarray(mask).save(masks/(source.stem+'.png'))
        feature_mask=cv2.erode((mask>250).astype('u1')*255,np.ones((3,3),'u1'));Image.fromarray(feature_mask).save(features/(name+'.png'))
        records[name]=dict(K=K.tolist(),image_size=[width,height],source=calibration_source,undistorted=bool(np.any(D)),rotation_degrees=rotation,calibration_fixed=bool(prior.get('fixed',True)) if prior else calibration_source.startswith('user_'),zero_distortion=bool(prior and not any(prior.get('distortion',[0]*5))))
        rows.append(dict(image=name,source=mode,fraction=float((mask>127).mean()),crop_original=box,original_size=transformed_size,source_original_size=list(original_size),prepared_size=[width,height],rotation_degrees=rotation,lens_prior=prior))
    (target/'intrinsics.yaml').write_text(yaml.safe_dump(dict(images=records)))
    if record:
        (job/'foreground.json').write_text(json.dumps(rows,indent=2)+'\n')
        (job/'camera_prior.json').write_text(json.dumps(dict(prior=prior,scope='Intrinsics and lens distortion only. No earlier camera poses, tracks or geometry initialize this run.'),indent=2)+'\n')
    return masks
