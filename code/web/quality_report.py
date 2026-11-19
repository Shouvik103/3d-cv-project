"""Summarize current uploaded reconstructions and retained attempts, without GT claims."""
from pathlib import Path
import argparse,json,csv


def load(p,default=None):return json.loads(p.read_text()) if p.exists() else default


def summarize(folder):
    job=load(folder/'job.json',{});state=load(folder/'state.json',{});sfm=load(folder/'reconstruction/summary.json',{});mesh=load(folder/'mesh/summary.json',{})
    return dict(method=job.get('method'),status=state.get('status'),registered=sfm.get('registered'),views=sfm.get('total_images'),sparse_points=sfm.get('points'),triangles=mesh.get('light',{}).get('faces'),fused_views=mesh.get('frames_fused'),depth=mesh.get('backend',job.get('depth_backend')),matching_size=job.get('max_side'),profile=job.get('quality_profile'),focal_prior=job.get('camera_prior'),reprojection=sfm.get('median_reprojection_px',sfm.get('mean_reprojection_px')),residual='median' if 'median_reprojection_px' in sfm else 'mean',seconds=state.get('elapsed_seconds'),camera_seconds=sfm.get('runtime_seconds'),mesh_seconds=mesh.get('timing_seconds',{}).get('total'),geometry_reused=job.get('reuse_geometry',False),peak_rss_mb=state.get('metrics',{}).get('peak_rss_mb'),warnings=state.get('warnings',[]),diagnostics=load(folder/'registration_diagnostics.json',{}))


def report(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];attempts=[]
    for folder in sorted(root.iterdir()):
        j=load(folder/'job.json',{})
        if j.get('method') not in ('ours','colmap') or j.get('is_comparison'):continue
        scene=j['name'].split(' · ')[0]
        if scene not in ('Car','Chair','Mic','Temple'):continue
        current=dict(scene=scene,id=folder.name,name=j['name'],**summarize(folder));rows.append(current)
        for p in sorted((folder/'attempts').glob('*')):
            if (p/'job.json').exists():attempts.append(dict(scene=scene,id=folder.name,attempt=p.name,**summarize(p)))
    doc=dict(scope='Development reconstructions, not a controlled accuracy benchmark. Car quality selection and camera calibration differ from earlier capture-bin runs. Intrinsic priors come from earlier COLMAP estimates, never GT poses or surfaces. No previous geometry initializes a fresh run. View count alone does not establish geometric accuracy. Partial and failed attempts remain visible.',rows=rows,attempts=attempts)
    (out/'report.json').write_text(json.dumps(doc,indent=2)+'\n')
    keys=['scene','method','status','registered','views','sparse_points','fused_views','triangles','depth','profile','matching_size','camera_seconds','mesh_seconds','geometry_reused','id']
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Quality campaign results','',doc['scope'],'','| Object / run | Method | Cameras | Depth | Fused views | Status |','|---|---|---:|---|---:|---|']
    for r in rows:lines.append(f"| {r['name']} | {r['method']} | {r['registered']}/{r['views']} | {r['depth']} | {r['fused_views']} | {r['status']} |")
    lines+=['','Retained attempts: '+str(len(attempts))+'. See report.json for measured per-view support, calibration sources, warnings, and timings.','', 'The 115,000-point browser limit is a display cap. It neither fixes the reconstructed point count nor fills missing surfaces.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return doc


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=Path('outputs/web_ui_extended'));p.add_argument('--out',type=Path,default=Path('validation/web_datasets_60'));a=p.parse_args();r=report(a.data_dir,a.out);print(len(r['rows']),'runs,',len(r['attempts']),'retained attempts')
