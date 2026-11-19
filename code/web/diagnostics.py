"""Measured registration diagnostics; image sharpness is context, not a diagnosis."""
from pathlib import Path
import json,sqlite3
import cv2
import numpy as np
import yaml


def read(path,default):
    return json.loads(path.read_text()) if path.exists() else default


def inspect_registration(job,summary):
    job=Path(job);poses=read(job/'reconstruction/poses.json',{})
    names=poses.get('image_names',[]);registered=set(poses.get('registered',[]));missing=[n for n in names if n not in registered]
    matches=read(job/'reconstruction/matching_report.json',{})
    edges=matches.get('verified_pairs',[])
    database=job/'reconstruction/database.db'
    if database.exists():
        with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
            lookup=dict(db.execute('SELECT image_id,name FROM images'))
            for pair,count in db.execute('SELECT pair_id,rows FROM two_view_geometries WHERE rows>0'):
                a,b=divmod(pair,2147483647)
                if a in lookup and b in lookup:edges.append(dict(images=[lookup[a],lookup[b]],inliers=count))
    graph={n:set() for n in names}
    for edge in edges:
        a,b=edge['images']
        if a in graph and b in graph:graph[a].add(b);graph[b].add(a)
    components=[];unseen=set(names)
    while unseen:
        pending=[min(unseen)];component=set()
        while pending:
            name=pending.pop()
            if name in component:continue
            component.add(name);pending.extend(graph[name]-component)
        unseen-=component;components.append(sorted(component))
    components.sort(key=len,reverse=True)
    checks=read(job/'reconstruction/registration_checks.json',{})
    rows=[]
    for name in names:
        links=[e['inliers'] for e in edges if name in e['images'] and any(n in registered and n!=name for n in e['images'])]
        path=job/'input/images'/name;photo=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        sharp=None
        if photo is not None:
            scale=min(1,512/max(photo.shape));photo=cv2.resize(photo,None,fx=scale,fy=scale)
            sharp=float(cv2.Laplacian(photo,cv2.CV_32F).var())
        rows.append(dict(image=name,registered=name in registered,verified_links_to_registered=len(links),strongest_verified_inliers=max(links,default=0),laplacian_variance_at_512=sharp,**checks.get(name,{})))
    calibration=yaml.safe_load((job/'input/intrinsics.yaml').read_text()) if (job/'input/intrinsics.yaml').exists() else {}
    record=calibration.get('images',{})
    insufficient=sum(row.get('reason')=='Insufficient shared 3D tracks' for row in rows if not row['registered'])
    weak=sum(row['verified_links_to_registered']<2 for row in rows if not row['registered'])
    explanation='All cameras reconstructed.'
    if missing:
        if checks:explanation=f'{insufficient} missing views lacked enough shared 3D tracks; {len(missing)-insufficient} failed PnP pose checks.'
        else:explanation=f'{weak} of the {len(missing)} missing views have fewer than two verified links to the selected model. {summary.get("models",1)} separate COLMAP model(s) were produced.'
        if len(components)>1:explanation+=f' The verified matching graph contains {len(components)} components.'
        explanation+=' Open Registration diagnostics for measured per-view support; sharpness alone does not establish the cause.'
    result=dict(registered=len(registered),total=len(names),missing=missing,explanation=explanation,
        verified_graph_available=bool(edges),checked_pairs=matches.get('checked_pairs'),verified_pairs=len(edges),components=components,views=rows,
        models=summary.get('model_images'),calibration_sources=sorted(set(r.get('source','unknown') for r in record.values())),
        initial_focal_range=[float(min(r['K'][0][0] for r in record.values())),float(max(r['K'][0][0] for r in record.values()))] if record else None,
        scope='Feature links and track support are measured. Sharpness depends on object texture and viewpoint; it is not a calibrated blur threshold. No ground-truth pose or geometry is used.')
    (job/'registration_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
