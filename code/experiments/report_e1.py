"""Generate E1 comparison figures and readable findings from measured JSONs."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from experiments.protocol import ROOT
from experiments.run_e1 import cloud_agreements,summarize
from eval.metrics import centers
from eval.align import transform_points

COLORS=dict(ours='#247b69',colmap='#b17a2e',vggt='#5567a5')


def report(root):
    root=Path(root);config=yaml.safe_load((root/'protocol.yaml').read_text())
    cloud_agreements(root,config);rows=summarize(root,config)
    target=root/'E1/figures';target.mkdir(exist_ok=True)
    scenes=list(config['scenes']);methods=list(COLORS)
    fig,axes=plt.subplots(3,len(scenes),figsize=(13,9),constrained_layout=True)
    metrics=[('registered','Registered images'),('ate_percent_extent','Aligned ATE (% trajectory extent)'),('seconds','End-to-end time (seconds)')]
    for col,scene in enumerate(scenes):
        selected={row['method']:row for row in rows if row['scene']==scene}
        for row_index,(metric,title) in enumerate(metrics):
            ax=axes[row_index,col]
            for i,method in enumerate(methods):
                value=selected[method][metric]
                if value is None:
                    ax.text(i,.03,'Pending / failed',rotation=90,ha='center',va='bottom',transform=ax.get_xaxis_transform(),fontsize=8,color='#8b554a')
                else:
                    ax.bar(i,value,color=COLORS[method],width=.6)
                    ax.annotate(f'{value:.2f}' if metric!='registered' else f'{int(value)}/{selected[method]["total"]}',(i,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(3),methods);ax.set_ylabel(title);ax.set_ylim(0,max([r[metric] or 0 for r in selected.values()]+[1])*1.25)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if row_index==0:ax.set_title(scene+'\n'+('GT camera errors' if config['scenes'][scene]['reference_kind']=='gt' else 'COLMAP agreement; no GT'))
    fig.savefig(target/'comparison.png',dpi=180);fig.savefig(target/'comparison.pdf');plt.close(fig)
    fig=plt.figure(figsize=(14,4.5),layout='constrained')
    for index,scene in enumerate(scenes,1):
        ax=fig.add_subplot(1,3,index,projection='3d')
        reference=json.loads((root/'datasets'/scene/'reference_poses.json').read_text())
        ref=centers(reference,reference['registered'])
        ax.plot(*ref.T,color='#777777',linestyle='--',marker='.',label='Reference')
        for method in methods:
            base=root/'E1'/scene/method
            if not (base/'evaluation.json').exists():continue
            evaluation=json.loads((base/'evaluation.json').read_text());metrics=evaluation.get('metrics')
            if not metrics:continue
            pose=json.loads((base/'poses.json').read_text());names=[n for n in reference['image_names'] if n in pose['registered']]
            alignment=metrics['alignment']
            C=transform_points(centers(pose,names),alignment['scale'],np.array(alignment['rotation']),np.array(alignment['translation']))
            ax.plot(*C.T,marker='.',color=COLORS[method],label=f'{method} {len(names)}/{len(reference["image_names"])}',linewidth=1)
        ax.set_title(scene);ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');ax.legend(fontsize=7)
    fig.suptitle('Camera trajectories after Sim(3); reference-frame arbitrary units')
    fig.savefig(target/'camera_trajectories.png',dpi=180);fig.savefig(target/'camera_trajectories.pdf');plt.close(fig)
    def number(value,digits=3):return '—' if value is None else f'{value:.{digits}f}'
    lines=['# E1 — matched-input comparison','',
           'These are recorded development results. Car/fountain pose numbers measure agreement with an existing full-data COLMAP pseudo-reference; only Temple has supplied GT cameras.',
           '', '| Scene | Method | Status | Registered | ATE % | AUC@5 | Median reprojection px | Time s | Peak RSS MB |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {row['scene']} | {row['method']} | {row['status']} | {row['registered'] if row['registered'] is not None else '—'}/{row['total']} | {number(row['ate_percent_extent'])} | {number(row['auc5'])} | {number(row['median_reprojection_px'])} | {number(row['seconds'],1)} | {number(row['peak_rss_mb'],1)} |")
    lines += ['', '## Interpretation and scope','',
              '- Missing registrations count as failures in pose AUC. ATE only evaluates shared registered cameras; registration rate must be read beside it.',
              '- The car subset spans the existing curated views. Both classical algorithms can split/lose coverage; low image reprojection residuals do not demonstrate globally correct camera geometry.',
              '- Our relaxed configuration was previously tuned on Temple; car selection previously used COLMAP poses. This is not an unseen generalization benchmark.',
              '- Our SfM and COLMAP share fixed calibration. VGGT receives the same photo bytes but uses its own 518px preprocessing and predicted intrinsics.',
              '- VGGT timing is local Apple MPS when that device is recorded, not a claimed CUDA/T4 result. End-to-end and forward-only timings have different scope.',
              '- Cloud agreement is measured against COLMAP sparse points, not ground-truth surface geometry. Pixel reprojection error is omitted for VGGT because it does not export observed feature tracks.',
              '', '## Reproduction','', '```bash',
              '.venv/bin/python code/experiments/run_e1.py',
              '.venv/bin/python code/experiments/report_e1.py','```','',
              'Frozen protocol: `protocol.yaml`. Image identities: `datasets/*/inputs.json`. Per-method logs, performance and evaluation: `E1/<scene>/<method>/`. Figures: `E1/figures/`.']
    (root/'E1/RESULTS.md').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    args=parser.parse_args();report(args.out)
