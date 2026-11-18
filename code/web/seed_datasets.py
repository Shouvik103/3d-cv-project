"""Create fresh, same-input comparisons from each object in datasets/, up to 60 views."""
from pathlib import Path
import argparse,hashlib,re,sys
import cv2,numpy as np,yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from web.seed_comparisons import request
from runtime import write_json
ROOT=Path(__file__).resolve().parents[2]


def natural(path):return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)',path.name)]


def select_photos(folder,limit=60):
    photos=sorted((p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}),key=natural)
    if len(photos)<=limit:return photos
    # One sharp image per capture-order bin, rather than the first 60 or a global
    # sharpness ranking that could discard an entire side of the object.
    selected=[]
    for indices in np.array_split(np.arange(len(photos)),limit):
        scored=[]
        for i in indices:
            image=cv2.imread(str(photos[i]),cv2.IMREAD_GRAYSCALE)
            if image is None:continue
            scale=min(1,512/max(image.shape));image=cv2.resize(image,None,fx=scale,fy=scale)
            scored.append((float(cv2.Laplacian(image,cv2.CV_32F).var()),int(i)))
        if scored:selected.append(photos[max(scored)[1]])
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8766');p.add_argument('--datasets',type=Path,default=ROOT/'datasets');p.add_argument('--out',type=Path,default=ROOT/'outputs/web_ui_extended/datasets60_manifest.json');p.add_argument('--objects',nargs='*');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.out.exists() and not a.dry_run:raise ValueError('Manifest exists; choose a new --out to deliberately start fresh runs')
    records=[]
    for folder in sorted(a.datasets.iterdir()):
        if not folder.is_dir() or (a.objects and folder.name not in a.objects):continue
        photos=select_photos(folder)
        if len(photos)<3:continue
        settings=dict(name=f'{folder.name.title()} · {len(photos)} views · object quality',method='all',quality_profile='object',matching_ratio=.8,max_side=1200,depth_backend='mono',make_mesh=True,mesh_quality='balanced')
        calibration=next(folder.rglob('intrinsics.yaml'),None)
        if calibration:
            recordsK=yaml.safe_load(calibration.read_text())['images'];Ks=[recordsK[p.name]['K'] for p in photos]
            if not all(np.allclose(Ks[0],K) for K in Ks):raise ValueError('Dataset requires per-image K support')
            settings['camera_matrix']=Ks[0];settings['matching_ratio']=.85
        if a.dry_run:print(folder.name,len(photos),photos[0].name,photos[-1].name,'supplied K' if calibration else 'estimated K');continue
        job=request(a.url,'/api/jobs',settings,'POST');record=dict(scene=folder.name,id=job['id'],settings=settings,selection='One sharp photo per natural capture-order bin; all photos when fewer than 60. No GT poses used.',images=[]);records.append(record);write_json(a.out,records)
        for i,photo in enumerate(photos):
            raw=photo.read_bytes();request(a.url,f'/api/jobs/{job["id"]}/images/{i}?name={photo.name}',raw,'PUT')
            record['images'].append(dict(index=i,path=str(photo.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest()))
        result=request(a.url,f'/api/jobs/{job["id"]}/start',{},'POST');record['children']=result['children'];write_json(a.out,records)
        print(folder.name,len(photos),'views; new runs:',','.join(result['children']),flush=True)


if __name__=='__main__':main()
