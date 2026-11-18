"""Sequential M4 real-GT mesh evaluation and M5 controlled source comparison."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse
import csv
import json
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from experiments.protocol import ROOT,sha256,update_ledger
from experiments.run_ablation import require_previous
from runtime import write_json


def mesh_config():
    from mesh.run_mesh import merge
    return merge(yaml.safe_load((ROOT/'configs/mesh_default.yaml').read_text()),
                 yaml.safe_load((ROOT/'configs/mesh_mono_small.yaml').read_text()))


def run_method(root,method,out):
    config=mesh_config();config['depth_backend']='vggt' if method=='vggt' else 'mono'
    out.mkdir(parents=True,exist_ok=True);frozen=out/'config.yaml'
    if frozen.exists() and yaml.safe_load(frozen.read_text())!=config: raise ValueError('Frozen mesh configuration changed')
    frozen.write_text(yaml.safe_dump(config))
    inputs=root/'datasets/fountain_s2';source=root/'E1/fountain_s2'/method
    if method=='vggt': scene=source/'raw.npz'
    else:
        scene=root/'mesh_scenes'/method;scene.mkdir(parents=True,exist_ok=True)
        write_json(scene/'scene.json',dict(reconstruction=str(source),images=str(inputs/'images'),depth_cache=str(root/'mesh_depth_cache/fountain_s2')))
    write_json(out/'inputs.json',dict(scene='fountain-P11',method=method,scene_path=str(scene),
        input_manifest_sha256=sha256(inputs/'inputs.json'),poses_sha256=sha256(source/'poses.json'),
        raw_sha256=sha256(scene) if method=='vggt' else None,
        note='Same frozen E1 views; ours/COLMAP share image-only monocular depth cache. Predicted poses/landmarks/K are not replaced by GT.'))
    if not (out/'status.json').exists():
        command=[sys.executable,str(ROOT/'code/mesh/run_mesh.py'),'--scene',str(scene),'--out',str(out),'--config',str(frozen)]
        print('Meshing',method,flush=True)
        with (out/'process.log').open('wb') as stream:
            p=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                             env=dict(os.environ,PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='1'))
        if not (out/'status.json').exists(): write_json(out/'status.json',dict(status='failed',error=f'Process exited {p.returncode}'))
    return scene,json.loads((out/'status.json').read_text())


def as_cameras(poses):
    names=poses['image_names']
    return SimpleNamespace(names=names,registered=[n in poses['registered'] for n in names],
                           R=[np.array(poses['R'].get(n,np.eye(3))) for n in names],
                           t=[np.array(poses['t'].get(n,[0,0,0])) for n in names])


def evaluate_gt(root,method,out,scene_path):
    import open3d as o3d
    from mesh.adapters import load_scene
    from mesh.mesh_export import read_mesh
    from mesh.mesh_eval import camera_alignment,compare_meshes,mesh_statistics,sample_surface
    from mesh.rendering import Renderer,orbit_camera
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import colormaps
    import matplotlib.pyplot as plt
    mesh=read_mesh(out/'mesh_full.ply')
    # The original GT has 26M faces. Keep it exact for surface sampling, without
    # adding per-face normals or a multi-GB raycasting acceleration structure.
    reference=o3d.io.read_triangle_mesh(str(ROOT/'data/strecha/fountain.ply'))
    cameras=json.loads((root/'M4/gt_poses.json').read_text());estimate=load_scene(scene_path)
    scale,Q,b=camera_alignment(estimate,as_cameras(cameras))
    scores=compare_meshes(mesh,reference,samples=200000,seed=0,sim3=(scale,Q,b),crop=True)
    T=np.eye(4);T[:3,:3]=scale*Q;T[:3,3]=b
    aligned=deepcopy(mesh).transform(T);cropped=aligned.crop(reference.get_axis_aligned_bounding_box())
    extent=float(np.linalg.norm(reference.get_axis_aligned_bounding_box().get_extent()))
    scores.update(kind='gt',scene='fountain-P11',method=method,gt_diagonal=extent,
        accuracy_percent_diagonal=100*scores['accuracy_mean']/extent,completeness_percent_diagonal=100*scores['completeness_mean']/extent,
        alignment='Umeyama Sim(3) on shared predicted and supplied GT camera centers; no geometry ICP',
        transform=T.tolist(),crop='GT mesh axis-aligned bounding box',full_before_crop=mesh_statistics(mesh),after_crop=mesh_statistics(cropped),
        reference_sha256=sha256(ROOT/'data/strecha/fountain.ply'),evaluation='200,000 independent area-uniform surface samples in each direction; not the official EPFL visibility-aware scorer')
    write_json(out/'gt_evaluation.json',scores)
    from scipy.spatial import cKDTree
    gt_samples=sample_surface(reference,200000,1)
    distances=cKDTree(gt_samples).query(np.asarray(cropped.vertices))[0]
    error_mesh=deepcopy(cropped);error_mesh.vertex_colors=o3d.utility.Vector3dVector(colormaps['magma'](np.clip(distances/(.02*extent),0,1))[:,:3])
    o3d.io.write_triangle_mesh(str(out/'gt_error.ply'),error_mesh)
    # Use the aligned first input camera for corresponding output/GT viewpoints.
    name=cameras['registered'][0];K=np.asarray(cameras['K'][name])*1.;K[0]*=640/cameras['image_size'][name][0];K[1]*=426/cameras['image_size'][name][1]
    R=np.asarray(cameras['R'][name]);t=np.asarray(cameras['t'][name])
    panels=[]
    q=gt_samples@R.T+t;uv=q@K.T;uv=uv[:,:2]/np.maximum(q[:,2:],1e-9)
    valid=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<426)
    image=np.full((426,640,3),245,dtype='u1');xy=uv[valid].astype(int);image[xy[:,1],xy[:,0]]=135
    panels=[('Reconstruction',Renderer(cropped).render(K,R,t,640,426)['color']),
            ('GT surface (200k samples)',image),
            ('Sampled surface error (clipped at 2%)',Renderer(error_mesh).render(K,R,t,640,426)['color'])]
    for name,(_,pixels) in zip(['reconstruction','ground_truth','surface_error'],panels): Image.fromarray(pixels).save(out/(name+'.png'))
    renderer=Renderer(cropped)
    for mode in ['shaded','wireframe']: Image.fromarray(renderer.render(K,R,t,640,426,mode)['color']).save(out/(mode+'.png'))
    fig,axes=plt.subplots(1,3,figsize=(15,4.1),layout='constrained')
    for ax,(title,pixels) in zip(axes,panels): ax.imshow(pixels);ax.set_title(title);ax.axis('off')
    bar=fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,2),cmap='magma'),ax=axes[-1],fraction=.03,pad=.02)
    bar.set_label('Distance / GT diagonal (%)')
    fig.savefig(out/'gt_comparison.png',dpi=160);fig.savefig(out/'gt_comparison.pdf');plt.close(fig)
    return scores


def report(root,experiment,rows):
    folder=root/experiment
    write_json(folder/'results.json',rows)
    with (folder/'results.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines=[f'# {experiment} — fountain-P11 mesh study','',
        '| Source | Status | Fused views | Faces | Accuracy % | Completeness % | F@0.5% | F@1% | F@2% |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        def v(k): return '—' if row.get(k) is None else f'{row[k]:.3f}'
        lines.append(f"| {row['method']} | {row['status']} | {row.get('fused_views','—')} | {row.get('faces','—')} | {v('accuracy_percent_diagonal')} | {v('completeness_percent_diagonal')} | {v('f005')} | {v('f01')} | {v('f02')} |")
    lines+=['','## Protocol and limits','',
        '- Real original EPFL fountain GT triangle mesh and supplied camera files; hashes and downloads are recorded. Sim(3) fits shared camera centers only, followed by GT-bounding-box crop; no mesh ICP.',
        '- 200,000 area-uniform samples in each direction, seed 0. Accuracy is estimate→GT distance; completeness is GT→estimate distance (lower is better). F-score thresholds are fractions of GT diagonal. This is not the official EPFL visibility-aware evaluation.',
        '- GT is the original 26M-face mesh, sampled without geometry simplification. The displayed GT and vertex-error colors use those 200k reference samples, so distances include sampling approximation.',
        '- Completeness covers the full GT mesh, including scan regions that may not be visible in all input images or retained by the reconstruction ROI. No visibility mask is applied; this makes it a strict full-scan comparison.',
        '- Ours and COLMAP use the same pinned Depth Anything V2 Small model, 384px inference, image-only depth cache, affine depth alignment gates, filtering and TSDF parameters (voxel divisor 192). Each uses its own E1 estimated cameras and sparse landmarks. Consequently camera, landmark-alignment and ROI effects are coupled; this is not a pose-only causal experiment.',
        '- VGGT uses its own jointly predicted cameras, intrinsics and depth, official 518px preprocessing, and the same filtering/TSDF policy; its confidence gate is recorded. Different depth models/resolutions are a confound, not an isolated pose comparison.',
        '- Same three image names are withheld from fusion when all eleven views register. They participated in SfM/VGGT camera estimation; VGGT also jointly inferred their RGB/depth. These are fusion-held-out rendering diagnostics, not strictly unseen-image reconstruction tests.',
        '- PSNR/SSIM are measured only where geometry renders; coverage is reported so incomplete meshes cannot appear successful from a small good-looking patch. Meshes may be partial and non-watertight.',
        '- GT geometry is used only for evaluation, never for depth alignment, meshing, or selecting reconstruction settings. No parameters were tuned against its scores.',
        '', 'Source: [EPFL original fountain model and research-use notice](https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/modelFountain.html). Cite Strecha et al., *On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery*, CVPR 2008.']
    for row in rows:
        if row.get('error'): lines.append(f"\n{row['method']} failure: {row['error']}")
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def collect(root,method,out,scene,status):
    row=dict(method=method,status=status['status'],fused_views=None,faces=None,accuracy_percent_diagonal=None,
        completeness_percent_diagonal=None,f005=None,f01=None,f02=None,coverage=None,psnr_db=None,ssim=None,seconds=None,error=status.get('error'))
    if status['status']=='complete':
        score=evaluate_gt(root,method,out,scene);summary=json.loads((out/'summary.json').read_text())
        from viz.view_mesh_web import export_viewer
        export_viewer(out/'mesh_light.ply',out/'interactive_mesh.html',
                      title=f'Fountain-P11: {method} mesh',poses=root/'E1/fountain_s2'/method/'poses.json')
        if method=='vggt':
            summary['heldout']['scope']='RGB and predicted depth from all views participated in joint VGGT inference; held-out maps excluded only from filtering/fusion. Not an unseen-image test.'
            write_json(out/'summary.json',summary)
        held=summary['heldout']['rows']
        def mean(k):
            values=[v[k] for v in held if v[k] is not None];return float(np.mean(values)) if values else None
        row.update(fused_views=summary['frames_fused'],faces=summary['full']['faces'],
            accuracy_percent_diagonal=score['accuracy_percent_diagonal'],completeness_percent_diagonal=score['completeness_percent_diagonal'],
            f005=score['thresholds']['0.005']['f_score'],f01=score['thresholds']['0.01']['f_score'],f02=score['thresholds']['0.02']['f_score'],
            coverage=mean('coverage'),psnr_db=mean('psnr_db'),ssim=mean('ssim'),seconds=summary['timing_seconds']['total'])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=['M4','M5'],required=True)
    p.add_argument('--out',type=Path,default=ROOT/'results/experiments_20261118');args=p.parse_args();root=args.out.resolve()
    require_previous(root,args.experiment);update_ledger(root,args.experiment,'running');(root/args.experiment).mkdir(exist_ok=True)
    if args.experiment=='M4':
        from data.download_strecha import download,reference_poses
        provenance=download(ROOT/'data/strecha');write_json(root/'M4/gt_provenance.json',provenance)
        names=json.loads((root/'E1/fountain_s2/ours/poses.json').read_text())['image_names']
        cameras,checks=reference_poses(ROOT/'data/strecha',names)
        write_json(root/'M4/gt_poses.json',cameras);write_json(root/'M4/camera_convention_check.json',checks)
        out=root/'M4/ours';scene,status=run_method(root,'ours',out);rows=[collect(root,'ours',out,scene,status)]
    else:
        rows=json.loads((root/'M4/results.json').read_text())
        write_json(root/'M5/reuse.json',dict(source='M4/ours',reason='Identical ours mesh; no re-meshing or selection by GT scores'))
        for method in ['colmap','vggt']:
            out=root/'M5'/method;scene,status=run_method(root,method,out);rows.append(collect(root,method,out,scene,status));report(root,'M5',rows)
        # Same GT camera, pixels, background and crop for the side-by-side figure.
        from PIL import Image,ImageDraw,ImageFont
        from matplotlib.font_manager import findfont
        font=ImageFont.truetype(findfont('DejaVu Sans'),24)
        sheet=Image.new('RGB',(1920,932),(245,245,245));draw=ImageDraw.Draw(sheet)
        for i,method in enumerate(['ours','colmap','vggt']):
            out=root/('M4' if method=='ours' else 'M5')/method
            for j,mode in enumerate(['reconstruction','shaded']):
                path=out/(mode+'.png')
                if path.exists(): sheet.paste(Image.open(path),(i*640,j*466+40))
            draw.text((i*640+12,8),method,fill=(25,35,30),font=font)
            draw.text((i*640+12,474),method+' - shaded geometry',fill=(25,35,30),font=font)
        sheet.save(root/'M5/side_by_side.png')
    report(root,args.experiment,rows)
    update_ledger(root,args.experiment,'completed' if all(r['status']=='complete' for r in rows) else 'limited',
                  results=f'{args.experiment}/results.csv',failed_attempts=sum(r['status']!='complete' for r in rows))
    print(args.experiment,'addressed',flush=True)


if __name__=='__main__':main()
