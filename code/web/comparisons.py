"""Same-upload comparisons; COLMAP pose agreement is explicitly not GT accuracy."""
from pathlib import Path
import csv,hashlib,json,threading
from runtime import write_json
from web.settings import METHOD_NAMES

LOCK=threading.RLock()


def build_comparison(folder):
    folder=Path(folder);metadata=json.loads((folder/'job.json').read_text())
    if not metadata.get('is_comparison'):raise ValueError('This run is not a three-method comparison')
    signature=[("schema",2)];jobs=[]
    for child in metadata['children']:
        p=folder.parent/child
        for name in ['job.json','state.json','reconstruction/poses.json']:
            f=p/name;signature.append((child,name,f.stat().st_mtime_ns if f.exists() else None))
        if (p/'job.json').exists():
            jobs.append((p,json.loads((p/'job.json').read_text()),json.loads((p/'state.json').read_text())))
    digest=hashlib.sha256(repr(signature).encode()).hexdigest()
    with LOCK:
        cached=folder/'comparison.json'
        if cached.exists():
            old=json.loads(cached.read_text())
            if old.get('signature')==digest:return old
        rows=[];reference=next((p for p,m,s in jobs if m.get('method')=='colmap' and (p/'reconstruction/poses.json').exists()),None)
        for p,m,s in jobs:
            metrics=s.get('metrics',{});row=dict(id=p.name,method=m['method'],name=m['name'],status=s['status'],progress=s.get('progress',0),prior_attempts=m.get('prior_attempts',0),depth_backend=metrics.get('depth_backend'),
                inference_partially_reused=metrics.get('inference_partially_reused',False),geometry_reused=metrics.get('geometry_reused',False),cloud_scope=metrics.get('cloud_scope'),registered=metrics.get('registered'),total=len(m['uploads']),points=metrics.get('dense_points',metrics.get('sparse_points')),
                triangles=metrics.get('triangles'),inference_mode=metrics.get('inference_mode','Single scene'),reprojection_px=metrics.get('reprojection_px'),reprojection_statistic=metrics.get('reprojection_statistic'),
                seconds=s.get('elapsed_seconds'),reconstruction_seconds=metrics.get('reconstruction_seconds'),mesh_seconds=metrics.get('mesh_seconds'),
                peak_rss_mb=metrics.get('peak_rss_mb'),watertight=metrics.get('watertight'),
                ate_percent_extent=None,auc5=None,reference='colmap' if reference else None,
                error=s.get('message') if s['status'] in ('failed','cancelled','interrupted') else None,
                cloud_url=f'/api/jobs/{p.name}/files/cloud_view' if s.get('artifacts',{}).get('cloud_view') else None,
                mesh_url=f'/api/jobs/{p.name}/files/mesh_view' if s.get('artifacts',{}).get('mesh_view') else None)
            camera_seconds=metrics.get('original_reconstruction_seconds',metrics.get('reconstruction_seconds'))
            if metrics.get('inference_partially_reused'):camera_seconds=None
            row['camera_solve_seconds']=camera_seconds
            row['pipeline_seconds']=camera_seconds+metrics['mesh_seconds'] if camera_seconds is not None and metrics.get('mesh_seconds') is not None else None
            if reference and (p/'reconstruction/poses.json').exists():
                try:
                    from eval.metrics import evaluate_poses
                    poses=json.loads((p/'reconstruction/poses.json').read_text());ref=json.loads((reference/'reconstruction/poses.json').read_text())
                    metric=evaluate_poses(poses,ref,allow_partial=True)
                    row.update(ate_percent_extent=metric['ate_percent_extent'],auc5=metric['auc']['5'])
                except (ValueError,KeyError) as exc:row['alignment_note']=str(exc)
            rows.append(row)
        output=dict(id=folder.name,name=metadata['name'],signature=digest,rows=rows,input_photos=len(metadata['uploads']),
                    missing_children=len(metadata['children'])-len(jobs),
                    scope='Same uploaded images and preparation settings. Classical methods share supplied/estimated K; VGGT predicts its own after official 518px preprocessing (padding for Object quality, cropping for Standard). Cameras and depth models differ.',
                    reference_scope='Camera ATE/AUC measure agreement with this upload’s COLMAP reconstruction, not ground truth. Missing camera pairs are penalized; undefined alignment is unavailable.',
                    timing_scope='Total job seconds include preparation, camera/depth inference, mesh fusion and browser export. Camera + mesh time combines the original camera solve and current mesh stage, including for mesh-only rebuilds. Last-attempt seconds include only that attempt. The serial queue runs one method at a time. Partially reused VGGT window refreshes have no comparable fresh camera + mesh timing; their last-attempt and original window records remain available. Hardware and caches affect timings.')
        write_json(cached,output)
        columns=['method','status','registered','total','points','triangles','inference_mode','reprojection_px','reprojection_statistic','seconds','pipeline_seconds','geometry_reused','inference_partially_reused','camera_solve_seconds','reconstruction_seconds','mesh_seconds','peak_rss_mb','watertight','ate_percent_extent','auc5','reference','error']
        with (folder/'comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
        render_plot(folder,rows)
        return output


def render_plot(folder,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#247b69','#bc842e','#7582d5'];fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,(key,title) in zip(axes.ravel(),[('registered','Registered / predicted cameras'),('pipeline_seconds','Camera + current mesh time (s)'),('ate_percent_extent','Camera ATE (%), COLMAP agreement — not GT'),('triangles','Displayed mesh triangles')]):
        for i,row in enumerate(rows):
            value=row[key]
            if value is not None:ax.bar(i,value,color=colors[i%3],width=.6)
            else:ax.text(i,.02,'Unavailable',rotation=90,ha='center',va='bottom',transform=ax.get_xaxis_transform(),fontsize=8)
        ax.set_xticks(range(len(rows)),[METHOD_NAMES[r['method']] for r in rows]);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same-upload comparison · missing metrics are not zero')
    fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
