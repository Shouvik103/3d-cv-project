"""Sequential E2 (BA), E4 (detector), and E6 (matching thresholds) studies."""
from copy import deepcopy
from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from experiments.protocol import ROOT,ORDER,update_ledger
from experiments.run_e1 import evaluate
from runtime import write_json


def variants(experiment,config):
    values={}
    if experiment=='E2':
        for mode in ['none','local','local_global']:
            cfg=deepcopy(config);cfg['ba']['mode']=mode;values[mode]=cfg
    elif experiment=='E4':
        for detector in ['sift','rootsift','orb']:
            cfg=deepcopy(config);cfg['features']['detector']=detector
            cfg['features']['rootsift']=detector=='rootsift';values[detector]=cfg
    elif experiment=='E6':
        values['baseline']=deepcopy(config)
        for ratio in [.7,.8,.9]:
            cfg=deepcopy(config);cfg['matching']['ratio']=ratio;values[f'ratio_{ratio}']=cfg
        for threshold in [.75,3.]:
            cfg=deepcopy(config);cfg['matching']['ransac_thresh_px']=threshold;values[f'ransac_{threshold}']=cfg
    else:raise ValueError('Unsupported ablation')
    return values


def require_previous(root,experiment):
    ledger=json.loads((root/'experiment_status.json').read_text())
    previous=ORDER[ORDER.index(experiment)-1]
    if ledger['experiments'][previous]['status'] not in ('completed','limited'):
        raise ValueError(f'Finish/address {previous} before starting {experiment}')


def report(experiment,root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/experiment
    write_json(folder/'results.json',rows)
    with (folder/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=[f'# {experiment} — measured ablation','',
           'Same prepared images and reference cameras as E1. Settings vary only as named. Feature caches are warm where a matching feature configuration already exists.',
           '', '| Scene | Variant | Registered | ATE % | AUC@5 | Reprojection px | Seconds | Status |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        def value(k):return '—' if row[k] is None else f'{row[k]:.3f}'
        lines.append(f"| {row['scene']} | {row['variant']} | {row['registered']}/{row['total']} | {value('ate_percent_extent')} | {value('auc5')} | {value('reprojection_px')} | {value('seconds')} | {row['status']} |")
    lines += ['', 'Car and fountain references are COLMAP agreement, not GT accuracy. Temple has supplied GT camera poses. ATE only uses shared cameras; missing registrations are penalized in AUC. Failures are retained.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    scenes=list(dict.fromkeys(r['scene'] for r in rows))
    fig,axes=plt.subplots(2,len(scenes),figsize=(6*len(scenes),7),squeeze=False,layout='constrained')
    for j,scene in enumerate(scenes):
        selected=[r for r in rows if r['scene']==scene];labels=[r['variant'] for r in selected]
        for i,(key,title) in enumerate([('auc5','Pose AUC@5 (missing cameras penalized)'),('ate_percent_extent','Aligned ATE (% trajectory extent)')]):
            ax=axes[i,j]
            ax.bar(range(len(selected)),[r[key] or 0 for r in selected],color='#397c6c')
            for n,row in enumerate(selected):
                if row[key] is None:ax.text(n,.02,'Failed',transform=ax.get_xaxis_transform(),rotation=90,ha='center')
            ax.set_xticks(range(len(labels)),labels,rotation=30,ha='right');ax.set_ylabel(title);ax.set_title(scene)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(folder/'comparison.png',dpi=180);fig.savefig(folder/'comparison.pdf');plt.close(fig)
    if experiment=='E2':
        import numpy as np
        from eval.metrics import centers
        from eval.align import transform_points
        fig,axes=plt.subplots(1,len(scenes),figsize=(6*len(scenes),4),squeeze=False,layout='constrained')
        for j,scene in enumerate(scenes):
            ax=axes[0,j]
            reference=json.loads((root/'datasets'/scene/'reference_poses.json').read_text())
            names=reference['image_names'];C=centers(reference,names);extent=np.linalg.norm(np.ptp(C,axis=0))
            for row in [r for r in rows if r['scene']==scene]:
                base=folder/scene/row['variant'];path=base/'evaluation.json'
                if not path.exists():continue
                evaluation=json.loads(path.read_text());metrics=evaluation.get('metrics') or {};alignment=metrics.get('alignment')
                if not alignment:continue
                pose=json.loads((base/'poses.json').read_text());registered=[n for n in names if n in pose['registered']]
                predicted=transform_points(centers(pose,registered),alignment['scale'],np.array(alignment['rotation']),np.array(alignment['translation']))
                error=np.full(len(names),np.nan)
                for i,name in enumerate(registered):error[names.index(name)]=100*np.linalg.norm(predicted[i]-C[names.index(name)])/extent
                ax.plot(range(1,len(names)+1),error,marker='.',label=row['variant'])
            ax.set_title(scene);ax.set_xlabel('Photo order (gaps = unregistered)');ax.set_ylabel('Aligned camera error (% reference extent)');ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            if ax.get_legend_handles_labels()[0]:ax.legend()
        fig.savefig(folder/'camera_error_by_view.png',dpi=180);fig.savefig(folder/'camera_error_by_view.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['E2','E4','E6'],required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    args=parser.parse_args();root=args.out;experiment=args.experiment;require_previous(root,experiment)
    config=yaml.safe_load((root/'protocol.yaml').read_text());base=yaml.safe_load((ROOT/config['sfm_config']).read_text())
    settings=variants(experiment,base)
    scenes={'E2':['car_s1','temple_gt'],'E4':['car_s1','fountain_s2'],'E6':['car_s1']}[experiment]
    update_ledger(root,experiment,'running',variants=list(settings),scenes=scenes)
    rows=[]
    for scene in scenes:
        for variant,run_config in settings.items():
            folder=root/experiment/scene/variant;folder.mkdir(parents=True,exist_ok=True)
            path=folder/'variant.yaml'
            if path.exists() and yaml.safe_load(path.read_text())!=run_config:
                raise ValueError('Variant configuration changed; use a new output directory')
            path.write_text(yaml.safe_dump(run_config,sort_keys=False))
            if not (folder/'performance.json').exists():
                print(experiment,scene,variant,flush=True)
                command=[sys.executable,str(ROOT/'code/experiments/worker.py'),'--method','ours','--scene',str(root/'datasets'/scene),
                         '--out',str(folder),'--config',str(root/'protocol.yaml'),'--variant-config',str(path)]
                with (folder/'process.log').open('wb') as stream:
                    result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
                if not (folder/'performance.json').exists():
                    write_json(folder/'performance.json',dict(method='ours',status='failed',error=f'Process exited {result.returncode}'))
            performance=json.loads((folder/'performance.json').read_text())
            evaluation={}
            if (folder/'poses.json').exists() and performance['status']=='completed':
                evaluation=evaluate(root/'datasets'/scene,folder,config['scenes'][scene]['reference_kind'],performance)
            metrics=evaluation.get('metrics') or {}
            rows.append(dict(scene=scene,variant=variant,status=performance['status'],registered=evaluation.get('estimated_registered',0),
                total=config['scenes'][scene]['views'],reference=config['scenes'][scene]['reference_kind'],
                ate_percent_extent=metrics.get('ate_percent_extent'),auc5=(metrics.get('auc') or {}).get('5'),
                reprojection_px=evaluation.get('reprojection',{}).get('median_px'),seconds=performance.get('wall_seconds'),error=performance.get('error')))
            report(experiment,root,rows)
    update_ledger(root,experiment,'completed',results=f'{experiment}/results.csv',failed_attempts=sum(r['status']=='failed' for r in rows))
    print(experiment,'completed',flush=True)


if __name__=='__main__':main()
