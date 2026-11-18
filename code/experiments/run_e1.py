"""E1: identical-input ours/COLMAP/VGGT comparison on three frozen scenes."""
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from runtime import write_json
from experiments.protocol import ROOT,prepare_scene,sha256,update_ledger
from eval.metrics import evaluate_poses
from eval.metrics import cloud_metrics
from eval.align import transform_points
from sfm.io_utils import load_ply


def evaluate(scene, output, reference_kind, performance):
    reference=json.loads((scene/'reference_poses.json').read_text())
    estimate=json.loads((output/'poses.json').read_text())
    result=dict(method=performance['method'],reference_kind=reference_kind,
                performance=performance,input_sha256=sha256(scene/'inputs.json'),
                estimated_registered=len(estimate['registered']),total_images=len(reference['image_names']))
    try:
        result['metrics']=evaluate_poses(estimate,reference,allow_partial=True)
    except ValueError as exc:
        result.update(metrics=None,evaluation_error=str(exc))
    summary=output/'summary.json'
    if summary.exists():result['run']=json.loads(summary.read_text())
    # Only evaluate true pixel reprojection for algorithms exporting actual tracks.
    if (output/'observations.json').exists():
        import numpy as np
        from sfm.io_utils import load_ply
        from sfm.geometry import reprojection_error
        points=load_ply(output/'points.ply')[0]
        tracks=json.loads((output/'observations.json').read_text())
        errors=[reprojection_error(X,xy,np.array(estimate['K'][n]),np.array(estimate['R'][n]),np.array(estimate['t'][n]))
                for X,track in zip(points,tracks.values()) for n,xy in track.items()]
        result['reprojection']=dict(median_px=float(np.median(errors)),mean_px=float(np.mean(errors)),
                                    scope='Processed input pixels; VGGT has no matching observed tracks')
        result['mean_track_length']=float(np.mean([len(t) for t in tracks.values()]))
    write_json(output/'evaluation.json',result)
    return result


def summarize(root, config):
    rows=[]
    for scene,settings in config['scenes'].items():
        for method in ['ours','colmap','vggt']:
            path=root/'E1'/scene/method
            perf=json.loads((path/'performance.json').read_text()) if (path/'performance.json').exists() else {}
            result=json.loads((path/'evaluation.json').read_text()) if (path/'evaluation.json').exists() else {}
            metrics=result.get('metrics') or {}
            rows.append(dict(scene=scene,method=method,status=perf.get('status','pending'),reference=settings['reference_kind'],
                registered=result.get('estimated_registered'),total=settings['views'],
                ate_percent_extent=metrics.get('ate_percent_extent'),auc5=(metrics.get('auc') or {}).get('5'),
                auc10=(metrics.get('auc') or {}).get('10'),auc30=(metrics.get('auc') or {}).get('30'),
                median_rotation_deg=metrics.get('median_rotation_deg'),median_translation_deg=metrics.get('median_translation_deg'),
                median_reprojection_px=result.get('reprojection',{}).get('median_px'),
                points=result.get('points'),mean_track_length=result.get('mean_track_length'),
                cloud_chamfer_agreement=result.get('pointcloud_agreement_to_colmap',{}).get('chamfer'),
                cloud_fscore_agreement=result.get('pointcloud_agreement_to_colmap',{}).get('fscore'),
                seconds=perf.get('wall_seconds'),peak_rss_mb=perf.get('peak_process_rss_mb'),error=perf.get('error')))
    write_json(root/'E1/results.json',rows)
    with (root/'E1/results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return rows


def cloud_agreements(root, config):
    import numpy as np
    for scene in config['scenes']:
        base=root/'E1'/scene;reference=base/'colmap'
        if not (reference/'evaluation.json').exists():continue
        ref=json.loads((reference/'poses.json').read_text())
        points_ref=load_ply(reference/'points.ply')[0]
        for method in ['ours','colmap','vggt']:
            folder=base/method;path=folder/'evaluation.json'
            if not path.exists():continue
            result=json.loads(path.read_text());points=load_ply(folder/'points.ply')[0]
            result['points']=len(points)
            try:
                pose=json.loads((folder/'poses.json').read_text())
                alignment=evaluate_poses(pose,ref)['alignment']
                aligned=transform_points(points,alignment['scale'],np.array(alignment['rotation']),np.array(alignment['translation']))
                agreement=cloud_metrics(aligned,points_ref,seed=config['seed'])
                agreement['scope']='Density-matched agreement with the matched-input COLMAP sparse cloud, NOT GT geometry accuracy'
                result['pointcloud_agreement_to_colmap']=agreement
            except ValueError as exc:result['cloud_agreement_error']=str(exc)
            write_json(path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/experiments.yaml')
    parser.add_argument('--methods',nargs='+',choices=['ours','colmap','vggt'],default=['ours','colmap','vggt'])
    parser.add_argument('--scenes',nargs='+')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text());args.out.mkdir(parents=True,exist_ok=True)
    saved=args.out/'protocol.yaml'
    if saved.exists() and yaml.safe_load(saved.read_text())!=cfg:
        raise ValueError('Benchmark configuration changed; use a new output root')
    saved.write_text(yaml.safe_dump(cfg,sort_keys=False))
    update_ledger(args.out,'E1','running',scope='Matched inputs, fixed classical intrinsics; full existing COLMAP model is a pseudo-reference on self-captured scenes')
    for key,scene in cfg['scenes'].items():
        if args.scenes and key not in args.scenes:continue
        data=args.out/'datasets'/key;prepare_scene(key,scene,cfg,data)
        for method in args.methods:
            out=args.out/'E1'/key/method
            if (out/'performance.json').exists():
                print('Retaining completed/failed attempt:',key,method,flush=True)
            else:
                out.mkdir(parents=True,exist_ok=True)
                print('Running E1:',key,method,flush=True)
                isolated=ROOT/'outputs/vggt_env/bin/python'
                interpreter=str(isolated) if method=='vggt' and isolated.exists() else sys.executable
                command=[interpreter,str(ROOT/'code/experiments/worker.py'),'--method',method,'--scene',str(data),'--out',str(out),'--config',str(saved)]
                with (out/'process.log').open('wb') as stream:
                    process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='1'))
                if not (out/'performance.json').exists():
                    write_json(out/'performance.json',dict(method=method,status='failed',error=f'Process exited {process.returncode} before a performance record was written'))
            perf=json.loads((out/'performance.json').read_text())
            if (out/'poses.json').exists() and perf['status']=='completed':evaluate(data,out,scene['reference_kind'],perf)
            cloud_agreements(args.out,cfg);summarize(args.out,cfg)
    cloud_agreements(args.out,cfg);rows=summarize(args.out,cfg)
    complete=all(r['status']=='completed' and r['ate_percent_extent'] is not None for r in rows)
    update_ledger(args.out,'E1','completed' if complete else 'partial',results='E1/results.csv',
                  notes='No GT camera accuracy is claimed for the car/fountain; those metrics are COLMAP agreement.')
    print('E1 status:', 'completed' if complete else 'partial',flush=True)
    return 0 if complete else 2


if __name__=='__main__':
    raise SystemExit(main())
