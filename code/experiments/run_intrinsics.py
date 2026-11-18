"""E5: approximate/calibrated/refined intrinsics on user-approved Temple GT."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import yaml
from experiments.protocol import ROOT,sha256,update_ledger
from experiments.run_ablation import require_previous,report
from experiments.run_e1 import evaluate
from runtime import write_json


def prepare_condition(source,target,approximate):
    calibrated=yaml.safe_load((source/'intrinsics.yaml').read_text())
    records=deepcopy(calibrated)
    for name,item in records['images'].items():
        if approximate:
            with Image.open(source/'images'/name) as image:width,height=image.size
            f=1.2*max(width,height)
            item.update(K=[[f,0,width/2],[0,f,height/2],[0,0,1]],source='fallback_1.2_max_side_no_EXIF')
    signature=dict(source_inputs_sha256=sha256(source/'inputs.json'),approximate=approximate,intrinsics=records)
    if (target/'inputs.json').exists():
        if json.loads((target/'inputs.json').read_text())!=signature:raise ValueError('Frozen E5 inputs changed')
        return
    target.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source/'images',target/'images')
    shutil.copy2(source/'reference_poses.json',target/'reference_poses.json')
    (target/'intrinsics.yaml').write_text(yaml.safe_dump(records))
    write_json(target/'inputs.json',signature)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118')
    args=parser.parse_args();root=args.out;require_previous(root,'E5')
    config=yaml.safe_load((root/'protocol.yaml').read_text());base=yaml.safe_load((ROOT/config['sfm_config']).read_text())
    update_ledger(root,'E5','running',scope='User chose calibrated public Temple; no car checkerboard calibration is available')
    rows=[]
    for label,approximate,refine in [('approximate',True,False),('calibrated',False,False),('approximate_refined',True,True)]:
        dataset=root/'datasets/E5'/label;prepare_condition(root/'datasets/temple_gt',dataset,approximate)
        folder=root/'E5/temple_gt'/label;folder.mkdir(parents=True,exist_ok=True)
        cfg=deepcopy(base);cfg['ba']['refine_focal']=refine
        variant=folder/'variant.yaml'
        if variant.exists() and yaml.safe_load(variant.read_text())!=cfg:raise ValueError('E5 configuration changed')
        variant.write_text(yaml.safe_dump(cfg,sort_keys=False))
        if not (folder/'performance.json').exists():
            print('E5',label,flush=True)
            command=[sys.executable,str(ROOT/'code/experiments/worker.py'),'--method','ours','--scene',str(dataset),'--out',str(folder),
                     '--config',str(root/'protocol.yaml'),'--variant-config',str(variant)]
            with (folder/'process.log').open('wb') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONUNBUFFERED='1'))
            if not (folder/'performance.json').exists():write_json(folder/'performance.json',dict(method='ours',status='failed',error=f'Process exited {result.returncode}'))
        performance=json.loads((folder/'performance.json').read_text());evaluation={};focal=None
        records=yaml.safe_load((dataset/'intrinsics.yaml').read_text())['images']
        initial=float(np.median([item['K'][0][0] for item in records.values()]))
        if performance['status']=='completed' and (folder/'poses.json').exists():
            evaluation=evaluate(dataset,folder,'gt',performance)
            pose=json.loads((folder/'poses.json').read_text());focal=float(np.median([K[0][0] for K in pose['K'].values()]))
        metrics=evaluation.get('metrics') or {}
        rows.append(dict(scene='temple_gt',variant=label,status=performance['status'],registered=evaluation.get('estimated_registered',0),
            total=config['scenes']['temple_gt']['views'],reference='gt',ate_percent_extent=metrics.get('ate_percent_extent'),
            auc5=(metrics.get('auc') or {}).get('5'),reprojection_px=evaluation.get('reprojection',{}).get('median_px'),
            seconds=performance.get('wall_seconds'),error=performance.get('error'),initial_fx=initial,final_fx=focal))
        report('E5',root,rows)
    note=['','## Calibration scope','',
          'The user approved Temple in place of unavailable real car checkerboard data. Calibrated means supplied Middlebury K, not a checkerboard calibration we performed.',
          'Approximate means f=1.2*max(width,height) with a centered principal point. The refined condition starts from the same approximate K and fits a shared fx/fy scale only during global BA after four cameras register. Principal point and distortion remain fixed. No GT focal value enters that optimization.',
          '', '| Condition | Initial median fx | Final median fx |','| --- | ---: | ---: |']
    for row in rows:note.append(f"| {row['variant']} | {row['initial_fx']:.3f} | {row['final_fx'] if row['final_fx'] is not None else 'Unavailable'} |")
    with (root/'E5/RESULTS.md').open('a') as stream:stream.write('\n'.join(note)+'\n')
    update_ledger(root,'E5','completed',results='E5/results.csv',scope='Calibrated public Temple substitution explicitly approved by user; car checkerboard study unavailable')
    print('E5 completed',flush=True)


if __name__=='__main__':main()
