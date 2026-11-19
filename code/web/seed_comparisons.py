"""Upload fresh Temple/car/chair comparisons through the normal HTTP job API."""
from pathlib import Path
import argparse,hashlib,json,sys,urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime import write_json

ROOT=Path(__file__).resolve().parents[2]


def request(base,path,data=None,method='GET'):
    headers={'X-Reconstruction-UI':'1'}
    if isinstance(data,dict):data=json.dumps(data).encode();headers['Content-Type']='application/json'
    with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers,method=method),timeout=120) as response:return json.load(response)


def datasets():
    import numpy as np
    temple=sorted((ROOT/'data/temple/images').glob('*.jpg'))
    car=json.loads((ROOT/'outputs/cars_curated_60/selection.json').read_text())['selected_images']
    car=sorted([item for item in car if item.get('target',{}).get('tier')=='oblique'],key=lambda item:item['target']['azimuth'])
    car=[ROOT/'outputs/cars_curated_60/calibrated_2400/images'/car[i]['name'] for i in np.linspace(0,len(car)-1,30,dtype=int)]
    chair=json.loads((ROOT/'outputs/chair_test/selection.json').read_text())['selected_images']
    chair=[ROOT/'nerf_synthetic/chair/test'/item['name'] for item in chair[::2]]
    for name,photos,K,note in [
        ('Temple',temple,[[1520.4,0,302.32],[0,1525.9,246.87],[0,0,1]],'All 16 public Middlebury Temple images. Supplied camera intrinsics only; no GT poses or geometry used for reconstruction.'),
        ('Car',car,[[1792.7160507828976,0,1200],[0,1792.7160507828976,900],[0,0,1]],'30 angularly spaced oblique views from the prior curated 60. Curation and input K previously used COLMAP estimates; fresh poses and geometry are computed independently.'),
        ('Chair',chair,[[1104.2233554111049,0,400],[0,1104.2233554111049,400],[0,0,1]],'30 spaced views from the prior selected 60, using original RGBA images. Input K is an earlier COLMAP estimate, not supplied GT. Original alpha is retained as a foreground mask.'),
    ]:
        yield name,photos,K,note


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8766');p.add_argument('--out',type=Path,default=ROOT/'outputs/web_ui_extended/seed_manifest.json');a=p.parse_args()
    if a.out.exists():raise ValueError('Seed manifest already exists; use a fresh filename to deliberately create new runs')
    records=[]
    for name,photos,K,note in datasets():
        settings=dict(name=name+' · fresh two-method comparison',method='all',camera_matrix=K,matching_ratio=.85 if name=='Temple' else .7,depth_backend='mono',max_side=1200,make_mesh=True,mesh_quality='balanced')
        job=request(a.url,'/api/jobs',settings,'POST');record=dict(scene=name,id=job['id'],settings=settings,note=note,images=[])
        records.append(record);write_json(a.out,records)
        for i,photo in enumerate(photos):
            raw=photo.read_bytes();request(a.url,f'/api/jobs/{job["id"]}/images/{i}?name={photo.name}',raw,'PUT')
            record['images'].append(dict(index=i,path=str(photo.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest()))
        result=request(a.url,f'/api/jobs/{job["id"]}/start',{},'POST');record['children']=result['children'];write_json(a.out,records)
        print(name,len(photos),'images; new runs:',','.join(result['children']),flush=True)


if __name__=='__main__':main()
