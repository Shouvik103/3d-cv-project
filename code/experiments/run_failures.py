"""E7: glossy/sparse car, controlled blur, and real low-texture Dino GT."""
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageFilter,ImageDraw,ImageFont
import yaml
from experiments.protocol import ROOT,sha256,prepare_scene,update_ledger
from experiments.run_ablation import require_previous
from experiments.run_e1 import evaluate
from runtime import write_json


def blurred_scene(source,target,sigma=4):
    if (target/'inputs.json').exists():return
    (target/'images').mkdir(parents=True,exist_ok=True)
    for p in sorted((source/'images').glob('*.jpg')):
        with Image.open(p) as image:image.filter(ImageFilter.GaussianBlur(sigma)).save(target/'images'/p.name,quality=95)
    for name in ['intrinsics.yaml','reference_poses.json']:shutil.copy2(source/name,target/name)
    write_json(target/'inputs.json',dict(source_inputs_sha256=sha256(source/'inputs.json'),
               intervention=f'Deliberate Gaussian blur, radius {sigma} px at prepared image resolution; geometry/K unchanged',
               prepared_sha256={p.name:sha256(p) for p in sorted((target/'images').glob('*.jpg'))}))


def report(root,rows,conditions):
    folder=root/'E7';write_json(folder/'results.json',rows)
    with (folder/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['# E7 — failure and robustness study','',
           'Real glossy car and deliberately sparse subsets reuse their exact E1/E3 measurements. Controlled blur is applied identically to every method, preserving calibrated camera geometry. Dino is the official real low-texture public scene.',
           '', '| Condition | Method | Registered | ATE % | AUC@5 | Status | Reason |',
           '| --- | --- | ---: | ---: | ---: | --- | --- |']
    for row in rows:
        def n(k):return '—' if row[k] is None else f'{row[k]:.3f}'
        reason=(row['error'] or '').replace('|','/')
        lines.append(f"| {row['condition']} | {row['method']} | {row['registered']}/{row['total']} | {n('ate_percent_extent')} | {n('auc5')} | {row['status']} | {reason} |")
    lines+=['','## Limits','',
            '- The car has a COLMAP pseudo-reference, not true GT. Its appearance is glossy, but these results do not isolate reflectivity as the only cause of failure.',
            '- Sparse car views use the existing five-image E3 subset, with substantial viewpoint gaps. No additional photos or poses were invented.',
            '- Blur is a controlled artificial degradation (Gaussian radius 4px), explicitly not a newly captured blurry scene. Temple GT camera poses remain valid.',
            '- Dino provides supplied GT camera poses. Its data were not used to tune our frozen relaxed configuration.',
            '- Failed reconstructions are retained with zero registration and zero AUC; undefined global alignment does not receive an invented ATE.',
            '- This study evaluates reconstruction outputs and geometry agreement; it does not claim success from attractive point-cloud appearance alone.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=list(conditions);fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    colors=dict(ours='#247b69',colmap='#b17a2e',vggt='#5567a5')
    for i,(method,color) in enumerate(colors.items()):
        selected={r['condition']:r for r in rows if r['method']==method}
        x=np.arange(len(labels))+(i-1)*.24
        axes[0].bar(x,[selected[n]['registered']/selected[n]['total'] if n in selected else 0 for n in labels],width=.24,label=method,color=color)
        axes[1].bar(x,[selected[n]['auc5'] or 0 if n in selected else 0 for n in labels],width=.24,label=method,color=color)
    for ax,title in zip(axes,['Registration rate','Pose AUC@5 (missing cameras penalized)']):
        ax.set_ylabel(title);ax.set_xticks(range(len(labels)),labels);ax.set_ylim(0,1.05);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.savefig(folder/'comparison.png',dpi=180);fig.savefig(folder/'comparison.pdf');plt.close(fig)
    font=ImageFont.truetype(matplotlib.font_manager.findfont('DejaVu Sans'),14)
    sheet=Image.new('RGB',(840,180*len(labels)),(246,247,244));draw=ImageDraw.Draw(sheet)
    for i,label in enumerate(labels):
        data=conditions[label]['data'];photos=sorted((data/'images').glob('*.jpg'))
        draw.text((12,i*180+8),label,fill=(30,45,35),font=font)
        for j,index in enumerate(np.linspace(0,len(photos)-1,4,dtype=int)):
            with Image.open(photos[int(index)]) as image:
                image=image.convert('RGB');image.thumbnail((196,130));sheet.paste(image,(j*210+7,i*180+34))
    sheet.save(folder/'input_conditions.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    args=parser.parse_args();root=args.out;require_previous(root,'E7')
    config=yaml.safe_load((root/'protocol.yaml').read_text())
    update_ledger(root,'E7','running')
    dino=ROOT/'data/middlebury/dinoSparseRing'
    if not (dino/'gt_poses.json').exists():
        from data.download_middlebury import download
        download('dinoSparseRing',ROOT/'data/middlebury')
    dino_data=root/'datasets/E7/dino_gt'
    spec=dict(images='data/middlebury/dinoSparseRing',intrinsics='data/middlebury/dinoSparseRing/intrinsics.yaml',
              reference='data/middlebury/dinoSparseRing/gt_poses.json',reference_kind='gt',views=16,
              notes='Official low-texture Dino sparse ring; no scene-specific parameter tuning')
    prepare_scene('dino_gt',spec,config,dino_data)
    blur_data=root/'datasets/E7/temple_blur';blurred_scene(root/'datasets/temple_gt',blur_data)
    conditions={
        'glossy_car':dict(data=root/'datasets/car_s1',reference='colmap',reuse=root/'E1/car_s1',total=20),
        'sparse_car_5':dict(data=root/'datasets/E3/car_s1_5',reference='colmap',reuse=root/'E3/car_s1/views_5',total=5),
        'blurred_temple':dict(data=blur_data,reference='gt',total=16),
        'low_texture_dino':dict(data=dino_data,reference='gt',total=16),
    }
    rows=[]
    for condition,settings in conditions.items():
        for method in ['ours','colmap','vggt']:
            folder=root/'E7'/condition/method
            if not (folder/'performance.json').exists():
                if settings.get('reuse'):
                    shutil.copytree(settings['reuse']/method,folder)
                    write_json(folder/'reuse.json',dict(source=str((settings['reuse']/method).relative_to(root)),reason='Exact same input images, calibration and method settings'))
                else:
                    folder.mkdir(parents=True,exist_ok=True)
                    python=str(ROOT/'outputs/vggt_env/bin/python') if method=='vggt' else sys.executable
                    command=[python,str(ROOT/'code/experiments/worker.py'),'--method',method,'--scene',str(settings['data']),
                             '--out',str(folder),'--config',str(root/'protocol.yaml')]
                    print('E7',condition,method,flush=True)
                    with (folder/'process.log').open('wb') as stream:
                        process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='1'))
                    if not (folder/'performance.json').exists():write_json(folder/'performance.json',dict(method=method,status='failed',error=f'Process exited {process.returncode}'))
            performance=json.loads((folder/'performance.json').read_text());evaluation={}
            if performance['status']=='completed' and (folder/'poses.json').exists():evaluation=evaluate(settings['data'],folder,settings['reference'],performance)
            metrics=evaluation.get('metrics') or {}
            rows.append(dict(condition=condition,method=method,status=performance['status'],registered=evaluation.get('estimated_registered',0),
                total=settings['total'],reference=settings['reference'],ate_percent_extent=metrics.get('ate_percent_extent'),
                auc5=(metrics.get('auc') or {}).get('5',0.),error=performance.get('error')))
            report(root,rows,conditions)
    update_ledger(root,'E7','completed',results='E7/results.csv',failed_attempts=sum(r['status']=='failed' for r in rows))
    print('E7 completed',flush=True)


if __name__=='__main__':main()
