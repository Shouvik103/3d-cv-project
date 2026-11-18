"""Original EPFL fountain-P11 GT assets; local research/evaluation use only."""
from pathlib import Path
import bz2
import json
import shutil
import tarfile
import urllib.request
import numpy as np

BASE = 'https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/'


def download(out):
    from experiments.protocol import sha256
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    assets = ['fountain.ply.bz2'] + [f'data/fountain_dense/urd/fountain_dense_{part}.tar.gz' for part in ['cameras','p','bounding']]
    records = []
    for asset in assets:
        path = out/Path(asset).name
        if not path.exists():
            with urllib.request.urlopen(BASE+asset, timeout=60) as src, path.with_suffix(path.suffix+'.part').open('wb') as dst:
                shutil.copyfileobj(src,dst)
            path.with_suffix(path.suffix+'.part').rename(path)
        records.append(dict(url=BASE+asset, file=path.name, sha256=sha256(path), bytes=path.stat().st_size))
        if asset.endswith('.tar.gz'):
            with tarfile.open(path) as archive: archive.extractall(out,filter='data')
    if not (out/'fountain.ply').exists():
        with bz2.open(out/'fountain.ply.bz2','rb') as src, (out/'fountain.ply').open('wb') as dst: shutil.copyfileobj(src,dst)
    provenance = dict(source=BASE+'modelFountain.html', use='Research and evaluation; acknowledge Strecha et al., CVPR 2008. Raw GT not redistributed in Git.',
                      assets=records, mesh_sha256=sha256(out/'fountain.ply'))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def read_camera(path):
    """Strecha stores camera-to-world rotation and camera center, not R|t."""
    values=np.fromstring(Path(path).read_text(),sep=' ')
    if len(values)!=26 or not np.isfinite(values).all(): raise ValueError('Expected 26 finite Strecha camera values')
    K=values[:9].reshape(3,3); R=values[12:21].reshape(3,3).T; center=values[21:24]
    if not np.allclose(R.T@R,np.eye(3),atol=1e-4) or np.linalg.det(R)<.999: raise ValueError('Invalid Strecha rotation')
    return K,R,-R@center,values[24:26].astype(int).tolist()


def reference_poses(directory,names):
    """Cross-check every convention conversion against the supplied P-matrix."""
    root=Path(directory)/'fountain_dense/urd'
    poses=dict(image_names=list(names),registered=list(names),K={},R={},t={},image_size={})
    checks={}
    for name in names:
        stem=Path(name).stem; K,R,t,size=read_camera(root/(stem+'.png.camera'))
        projection=K@np.column_stack([R,t]); supplied=np.loadtxt(root/(stem+'.png.P'))
        relative=float(np.linalg.norm(projection-supplied)/np.linalg.norm(supplied))
        if relative>1e-5: raise ValueError(f'Camera/P-matrix convention mismatch: {name}')
        checks[name]=relative
        for key,value in [('K',K),('R',R),('t',t)]: poses[key][name]=value.tolist()
        poses['image_size'][name]=size
    return poses,checks
