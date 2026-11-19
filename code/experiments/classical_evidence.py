"""Derive active two-method evidence without changing retained measurements.

Historical evidence is read only. Scalar reconstruction results are projected,
not rerun; aggregate figures and reports are regenerated for the active methods.
"""
from pathlib import Path
import argparse,csv,json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from runtime import write_json
from experiments.protocol import ROOT,ORDER,sha256
METHODS=('ours','colmap')


def projection(value):
    if isinstance(value,list):return [projection(v) for v in value if not isinstance(v,dict) or v.get('method','ours') in METHODS]
    if isinstance(value,dict):return {k:projection(v) for k,v in value.items() if k!='vggt'}
    if isinstance(value,str):return value.replace('Processed input pixels; VGGT has no matching observed tracks','Processed input pixels on observed feature tracks')
    return value


def figures(folder,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    if not rows:return
    keys=[('registered','Registered cameras'),('ate_percent_extent','Camera ATE (% extent)'),('auc5','Pose AUC@5'),('seconds','Recorded time (s)')]
    if folder.name.startswith('M'):keys=[('accuracy_percent_diagonal','Accuracy (% GT diagonal)'),('completeness_percent_diagonal','Completeness (% GT diagonal)'),('f01','F-score at 1% GT diagonal'),('coverage','Fusion-held-out render coverage')]
    fig,axes=plt.subplots(2,2,figsize=(14,max(7,1.8+len(rows)*.35)),layout='constrained')
    labels=[' / '.join(str(v) for v in [r.get('scene',r.get('condition','')),r.get('variant',r.get('views','')),r.get('method','ours')] if v!='') for r in rows]
    for ax,(key,title) in zip(axes.ravel(),keys):
        for i,r in enumerate(rows):
            v=r.get(key)
            if isinstance(v,(int,float)):ax.barh(i,v,color='#247b69' if r.get('method','ours')=='ours' else '#bc842e')
            else:ax.text(.01,i,'Unavailable',transform=ax.get_yaxis_transform(),va='center',fontsize=7)
        ax.set_yticks(np.arange(len(rows)),labels,fontsize=7);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(folder.name+' · retained classical measurements (not a new run)')
    fig.savefig(folder/'comparison.png',dpi=130);fig.savefig(folder/'comparison.pdf');plt.close(fig)


def derive(source,destination,renders=None):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or source in destination.parents:raise ValueError('Choose an output separate from historical source evidence')
    destination.mkdir(parents=True,exist_ok=True);sources=[]
    def record(p):sources.append(dict(path=str(p.relative_to(source)),sha256=sha256(p)))
    for p in source.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(source)
        if any(part.startswith(('vggt','E1_initial')) for part in rel.parts):continue
        # Aggregate reports/figures are regenerated, never copied with old legends.
        is_method=any(part in METHODS for part in rel.parts)
        if p.suffix.lower() in ('.png','.pdf','.html') and not is_method:continue
        if p.name in ('evidence_manifest.json','verification.json','environment_check.json') or p.suffix.lower()=='.md':continue
        if p.suffix.lower() not in ('.json','.yaml','.csv','.png','.pdf','.html'):continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);record(p)
        if p.suffix=='.json':write_json(target,projection(json.loads(p.read_text())))
        elif p.suffix=='.yaml':target.write_text(yaml.safe_dump(projection(yaml.safe_load(p.read_text())),sort_keys=False))
        elif p.suffix=='.csv':continue # Rewritten from projected result JSON below.
        else:shutil.copy2(p,target)
    for stage in ORDER:
        folder=destination/stage;folder.mkdir(exist_ok=True);path=folder/'results.json'
        if not path.exists():continue
        rows=json.loads(path.read_text());keys=list(dict.fromkeys(k for r in rows for k in r))
        with (folder/'results.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)
        columns=[k for k in ['scene','condition','variant','method','status','registered','total','views','ate_percent_extent','auc5','seconds','fused_views','accuracy_percent_diagonal','completeness_percent_diagonal','f01'] if k in keys]
        lines=[f'# {stage} — Our SfM / COLMAP','', 'Derived from retained 18 November 2026 development measurements. Only active methods are included; no reconstruction was rerun. Failures and unavailable values remain. Source hashes are in derivation.json.','', '| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for r in rows:lines.append('| '+' | '.join('—' if r.get(k) is None else str(r[k]) for k in columns)+' |')
        (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n');figures(folder,rows)
    if renders:
        # Scientific comparison panel from the original common-camera renders.
        from PIL import Image,ImageDraw
        panel=Image.new('RGB',(1280,972),'#f5f5f5');draw=ImageDraw.Draw(panel)
        for i,method in enumerate(METHODS):
            original=Path(renders)/('M4' if method=='ours' else 'M5')/method
            for j,name in enumerate(('reconstruction','shaded')):
                path=original/(name+'.png')
                if not path.exists():raise ValueError('Missing original aligned rendering: '+str(path))
                with Image.open(path) as im:panel.paste(im,(i*640,j*486+28))
                draw.text((i*640+12,j*486+7),method+' · '+name,fill='#182c24')
        panel.save(destination/'M5/side_by_side.png')
    write_json(destination/'derivation.json',dict(source_collection=source.name,methods=list(METHODS),source_files=sources,
        scope='Two-method projection of historical development evidence. Retained reconstruction metrics are unchanged; reports and aggregate figures are regenerated. Not a new reconstruction or timing experiment. Camera agreement is not GT accuracy. Original evidence is retained separately.'))
    (destination/'RESULTS.md').write_text('# Classical comparison evidence\n\nOur SfM and COLMAP only. See derivation.json for source hashes and scope; E1–E7 and M4/M5 contain measured tables.\n')
    manifest=[]
    for p in sorted(destination.rglob('*')):
        if p.is_file() and p.name!='evidence_manifest.json':manifest.append(dict(path=str(p.relative_to(destination)),sha256=sha256(p),bytes=p.stat().st_size))
    write_json(destination/'evidence_manifest.json',dict(methods=list(METHODS),source_directory=source.name,files=manifest,total_bytes=sum(r['bytes'] for r in manifest)))
    return len(manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'validation/experiments_20261118');p.add_argument('--out',type=Path,default=ROOT/'validation/experiments_20261119_classical');p.add_argument('--renders',type=Path,default=ROOT/'results/experiments_20261118');a=p.parse_args();print(derive(a.source,a.out,a.renders),'files')
