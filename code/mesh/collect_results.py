"""Mirror measured mesh summaries and selected figures without large raw outputs."""
from pathlib import Path
import argparse,csv,json,shutil


def collect(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for scene in sorted(root.iterdir()):
        if not scene.is_dir() or not (scene/'summary.json').exists():continue
        s=json.loads((scene/'summary.json').read_text());dest=out/scene.name;dest.mkdir(exist_ok=True)
        for n in ['summary.json','alignment.json','convention.json','config.yaml']:
            if (scene/n).exists():shutil.copy2(scene/n,dest/n)
        for n in ['colored_0.png','colored_1.png','shaded_0.png','wireframe_0.png','normals_0.png','render_manifest.json']:
            if (scene/'renders'/n).exists():shutil.copy2(scene/'renders'/n,dest/n)
        viewrows=s['heldout']['rows']
        def avg(k):
            a=[r[k] for r in viewrows if r[k] is not None];return sum(a)/len(a) if a else None
        rows.append(dict(scene=scene.name,backend=s['backend'],registered=s['registered_views'],fused=s['frames_fused'],full_vertices=s['full']['vertices'],full_faces=s['full']['faces'],light_faces=s['light']['faces'],boundary_edges=s['full']['boundary_edges'],watertight=s['full']['watertight'],heldout=len(viewrows),coverage=avg('coverage'),psnr_db=avg('psnr_db'),ssim=avg('ssim'),seconds=s['timing_seconds']['total']))
    with (out/'mesh_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'mesh_metrics.json').write_text(json.dumps(rows,indent=2))
    for experiment in ['ablations_sample','ablations_fixture','failure_study']:
        src=root/experiment
        if not src.exists():continue
        dst=out/experiment;dst.mkdir(exist_ok=True)
        for p in src.iterdir():
            if p.is_file() and p.suffix in ['.json','.png']:shutil.copy2(p,dst/p.name)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/mesh'));p.add_argument('--out',type=Path,default=Path('validation/mesh'));a=p.parse_args();collect(a.root,a.out)
