"""E3: fixed view-count subsets, identical across both methods."""
from copy import deepcopy
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from experiments.protocol import ROOT,prepare_scene,update_ledger
from experiments.run_e1 import evaluate
from experiments.run_ablation import require_previous
from runtime import write_json


def save_report(root,rows):
    folder=root/'E3';write_json(folder/'results.json',rows)
    with (folder/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['# E3 — number of input views','',
           'Uniformly spaced image subsets are frozen independently for each view count and shared by all methods. These cover the available trajectory; subsets are not claimed to be nested.',
           'The car has 5/10/20/30-view conditions. Temple has only 16 original images, so its GT conditions are 5/10/16; no 20/30-view GT results are invented.',
           'Byte-identical E1 inputs/configs reuse their original measured runs with provenance. Other conditions are fresh processes. Model weights are cached; feature-cache conditions are recorded in run folders.',
           '', '| Scene | Views | Method | Registered | ATE % | AUC@5 | Seconds | Status |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        def n(key):return '—' if row[key] is None else f'{row[key]:.3f}'
        lines.append(f"| {row['scene']} | {row['views']} | {row['method']} | {row['registered']}/{row['views']} | {n('ate_percent_extent')} | {n('auc5')} | {n('seconds')} | {row['status']} |")
    lines+=['','Camera error on the car is COLMAP agreement, not ground truth. Temple is a development GT scene. Failed runs retain their reason; their registration rate is zero. AUC on failed reconstructions is zero rather than omitted from the aggregate curve.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors=dict(ours='#247b69',colmap='#b17a2e')
    for j,scene in enumerate(['car_s1','temple_gt']):
        for method,color in colors.items():
            selected=sorted([r for r in rows if r['scene']==scene and r['method']==method],key=lambda r:r['views'])
            if not selected:continue
            x=[r['views'] for r in selected]
            axes[0,j].plot(x,[r['registered']/r['views'] for r in selected],marker='o',label=method,color=color)
            axes[1,j].plot(x,[r['auc5'] or 0 for r in selected],marker='o',label=method,color=color)
        for i in range(2):
            ax=axes[i,j];ax.set_title(scene);ax.set_xlabel('Input images');ax.grid(alpha=.2);ax.set_ylim(-.03,1.03)
            if ax.get_legend_handles_labels()[0]:ax.legend()
            ax.spines[['top','right']].set_visible(False)
        axes[0,j].set_ylabel('Registration rate');axes[1,j].set_ylabel('Pose AUC@5; failures included')
    fig.savefig(folder/'comparison.png',dpi=180);fig.savefig(folder/'comparison.pdf');plt.close(fig)


def refresh(root):
    """Rebuild evaluation/report files without rerunning any reconstruction."""
    config=yaml.safe_load((root/'protocol.yaml').read_text());rows=[]
    for scene,counts in [('car_s1',[5,10,20,30]),('temple_gt',[5,10,16])]:
        for count in counts:
            data=root/'datasets/E3'/f'{scene}_{count}'
            for method in ['ours','colmap']:
                folder=root/'E3'/scene/f'views_{count}'/method
                if not (folder/'performance.json').exists():continue
                performance=json.loads((folder/'performance.json').read_text());evaluation={}
                if performance['status']=='completed' and (folder/'poses.json').exists():
                    evaluation=evaluate(data,folder,config['scenes'][scene]['reference_kind'],performance)
                metrics=evaluation.get('metrics') or {}
                rows.append(dict(scene=scene,views=count,method=method,status=performance['status'],
                    registered=evaluation.get('estimated_registered',0),reference=config['scenes'][scene]['reference_kind'],
                    ate_percent_extent=metrics.get('ate_percent_extent'),auc5=(metrics.get('auc') or {}).get('5',0.),
                    seconds=performance.get('wall_seconds'),reused=bool((folder/'reuse.json').exists()),error=performance.get('error')))
    if rows:save_report(root,rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    args=parser.parse_args();root=args.out;require_previous(root,'E3')
    config=yaml.safe_load((root/'protocol.yaml').read_text())
    update_ledger(root,'E3','running',conditions=dict(car_s1=[5,10,20,30],temple_gt=[5,10,16]))
    rows=[]
    for scene,counts in [('car_s1',[5,10,20,30]),('temple_gt',[5,10,16])]:
        for count in counts:
            spec=deepcopy(config['scenes'][scene]);spec['views']=count
            data=root/'datasets/E3'/f'{scene}_{count}'
            record=prepare_scene(f'{scene}_{count}',spec,config,data)
            original=json.loads((root/'datasets'/scene/'inputs.json').read_text())
            can_reuse=record['prepared_sha256']==original['prepared_sha256']
            for method in ['ours','colmap']:
                folder=root/'E3'/scene/f'views_{count}'/method
                previous=root/'E1'/scene/method
                if not (folder/'performance.json').exists():
                    if can_reuse and (previous/'evaluation.json').exists():
                        shutil.copytree(previous,folder)
                        write_json(folder/'reuse.json',dict(source=str(previous.relative_to(root)),
                                   reason='Identical prepared image hashes, calibration, method settings and view count'))
                    else:
                        folder.mkdir(parents=True,exist_ok=True)
                        interpreter=sys.executable
                        command=[interpreter,str(ROOT/'code/experiments/worker.py'),'--method',method,'--scene',str(data),'--out',str(folder),'--config',str(root/'protocol.yaml')]
                        print('E3',scene,count,method,flush=True)
                        with (folder/'process.log').open('wb') as stream:
                            process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
                        if not (folder/'performance.json').exists():write_json(folder/'performance.json',dict(method=method,status='failed',error=f'Process exited {process.returncode}'))
                performance=json.loads((folder/'performance.json').read_text());evaluation={}
                if performance['status']=='completed' and (folder/'poses.json').exists():
                    evaluation=evaluate(data,folder,spec['reference_kind'],performance)
                metrics=evaluation.get('metrics') or {}
                rows.append(dict(scene=scene,views=count,method=method,status=performance['status'],
                    registered=evaluation.get('estimated_registered',0),reference=spec['reference_kind'],
                    ate_percent_extent=metrics.get('ate_percent_extent'),auc5=(metrics.get('auc') or {}).get('5',0.),
                    seconds=performance.get('wall_seconds'),reused=bool((folder/'reuse.json').exists()),error=performance.get('error')))
                save_report(root,rows)
    update_ledger(root,'E3','completed',results='E3/results.csv',failed_attempts=sum(r['status']=='failed' for r in rows))
    print('E3 completed',flush=True)


if __name__=='__main__':main()
