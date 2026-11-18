"""Add a mesh to an existing SfM scene; python code/mesh/run_mesh.py --help."""
from pathlib import Path
import argparse,copy,json,logging,sys,time,resource,importlib.metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,yaml
from runtime import manifest,write_json
from mesh.adapters import load_scene,check_convention
from mesh.depth_mono import MonoDepth
from mesh.align_depth import align_image
from mesh.filters import roi_bounds,filter_depth,consistency_filter
from mesh.tsdf_fusion import fuse
from mesh.mesh_clean import clean
from mesh.mesh_export import export_meshes
from mesh.mesh_eval import mesh_statistics,heldout_metrics

ROOT=Path(__file__).resolve().parents[2]

def merge(base,patch):
    out=copy.deepcopy(base)
    for k,v in patch.items():out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
    return out


def run(scene_path,out,config,images=None,backend=None,cache_only=False,progress=None):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    cfg=copy.deepcopy(config);method=backend or cfg['depth_backend'];cfg['depth_backend']=method
    cfg['cache_only']=bool(cache_only or cfg.get('cache_only',False))
    write_json(out/'run_manifest.json',manifest(cfg));(out/'config.yaml').write_text(yaml.safe_dump(cfg))
    if method not in ('mono','sgbm','vggt'):raise ValueError('Unknown depth backend')
    scene=load_scene(scene_path,images);check=check_convention(scene,out)
    if method=='vggt' and scene.vggt_path is None:raise ValueError('VGGT fusion requires its raw NPZ with its own poses, RGB, depth and confidence')
    registered=[i for i,v in enumerate(scene.registered) if v];rng=np.random.default_rng(cfg['heldout'].get('seed',cfg['seed']))
    n=min(cfg['heldout']['num_views'],max(0,len(registered)-3));heldout=sorted(rng.choice(registered,n,replace=False).tolist()) if n else [];training=[i for i in registered if i not in heldout]
    roi=roi_bounds(scene.points,cfg['filter']['roi_percentiles'],cfg['filter']['roi_margin_frac']);predictor=MonoDepth(cfg['mono']);frames=[];records=[];depth_time=0.;align_time=0.
    def notify(phase, done, total, message):
        if progress is not None:progress(phase,done,total,message)
    notify('depth',0,len(training),'Estimating stereo depth')
    for depth_index,i in enumerate(training,1):
        record=dict(image=scene.names[i],index=i)
        if method=='mono':record.update(s=None,b=None,n_inliers=0,median_rel_residual=None)
        try:
            tick=time.perf_counter()
            if method=='mono':
                relative,cache=predictor.predict(scene,i,cache_only or cfg.get('cache_only',False));depth_time+=time.perf_counter()-tick;tick=time.perf_counter()
                depth,alignment=align_image(scene,i,relative,cfg['align'],cfg['seed']+i);align_time+=time.perf_counter()-tick
                h,w=depth.shape;rgb=scene.image(i);K=scene.K[i].copy();K[0]*=w/rgb.shape[1];K[1]*=h/rgb.shape[0]
                frame=dict(index=i,color=cv2.resize(rgb,(w,h)),depth=depth,K=K,R=scene.R[i],t=scene.t[i]);record.update(alignment,cache=cache)
            elif method=='sgbm':
                from mesh.depth_sgbm import estimate
                frame,report=estimate(scene,i,training,cfg['sgbm']);depth_time+=time.perf_counter()-tick;record.update(report)
            else:
                with np.load(scene.vggt_path,allow_pickle=False) as d:
                    depth=np.squeeze(d['depth'][i]).astype('f4');confidence=np.squeeze(d['depth_conf'][i]) if 'depth_conf' in d else None
                    if confidence is not None:depth=np.where(confidence>=np.quantile(confidence,cfg['vggt']['confidence_quantile']),depth,0)
                h,w=depth.shape;rgb=scene.image(i);K=scene.K[i].copy();K[0]*=w/rgb.shape[1];K[1]*=h/rgb.shape[0]
                frame=dict(index=i,color=cv2.resize(rgb,(w,h)),depth=depth,K=K,R=scene.R[i],t=scene.t[i]);depth_time+=time.perf_counter()-tick
            frame['depth']=filter_depth(frame,roi,cfg['filter'],scene.mask(i));record['pixels_before_consistency']=int((frame['depth']>0).sum())
            if record['pixels_before_consistency']<30:raise ValueError('Too few valid foreground depth pixels')
            frames.append(frame);record['status']='accepted'
        except (ValueError,RuntimeError,OSError) as e:
            record.update(getattr(e,'details',{}));record.update(status='skipped',reason=str(e));logging.warning('%s: %s',scene.names[i],e)
        records.append(record);write_json(out/'alignment.json',dict(backend=method,images=records));logging.info('%s: %s',scene.names[i],record['status'])
        notify('depth',depth_index,len(training),f'Estimating depth · {depth_index}/{len(training)} views checked')
    if len(frames)<2:raise RuntimeError('Fewer than two depth maps passed the gates; inspect alignment.json or use --backend sgbm')
    notify('filtering',0,1,'Checking depth consistency across views')
    tick=time.perf_counter()
    if cfg['filter']['consistency']:frames=consistency_filter(frames,cfg['filter']['consistency_rel_tol'],cfg['filter']['consistency_min_views'])
    frames=[f for f in frames if (f['depth']>0).sum()>=30]
    if len(frames)<2:raise RuntimeError('Consistency rejected too much depth; inspect data or explicitly ablate consistency')
    sample_record=None
    if cfg.get('export_depth_samples',False):
        from mesh.depth_cloud import export_depth_samples
        sample_record=export_depth_samples(frames,out/'depth_samples.ply')
    filtering_time=time.perf_counter()-tick;voxel=scene.extent/cfg['tsdf']['voxel_div'];trunc=max(float(f['depth'].max()) for f in frames)*1.01
    notify('fusion',0,1,'Fusing depth maps into a surface')
    tick=time.perf_counter();mesh,volume=fuse(frames,voxel,cfg['tsdf']['sdf_trunc_voxels']*voxel,trunc);fusion_time=time.perf_counter()-tick
    notify('cleanup',0,1,'Cleaning and exporting triangle meshes')
    import open3d as o3d
    cloud=volume.extract_point_cloud()
    if cloud.has_colors():cloud.colors=o3d.utility.Vector3dVector(np.clip(np.asarray(cloud.colors),0,1))
    o3d.io.write_point_cloud(str(out/'dense.ply'),cloud)
    tick=time.perf_counter()
    if cfg['poisson']['enabled']:
        from mesh.poisson_mesh import reconstruct
        mesh=reconstruct(cloud,cfg['poisson']['depth'],cfg['poisson']['density_quantile'])
    full,light=clean(mesh,roi,cfg['clean']['min_component_frac'],cfg['clean']['light_target_triangles']);export_meshes(full,light,out,cfg['export']);extraction_time=time.perf_counter()-tick
    notify('mesh_preview',0,1,'Preparing mesh preview and measurements')
    held=heldout_metrics(full,scene,heldout)
    from mesh.rendering import Renderer,orbit_camera
    K,R,t=orbit_camera(light);rendered=Renderer(light).render(K,R,t,640,480);cv2.imwrite(str(out/'preview.png'),rendered['color'][:,:,::-1])
    timing=dict(depth=depth_time,alignment=align_time,filtering=filtering_time,fusion=fusion_time,clean_export=extraction_time,total=time.perf_counter()-start)
    memory=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    summary=dict(manifest(cfg),depth_samples=sample_record,tsdf_cloud_points=len(cloud.points),scene=str(Path(scene_path)),backend=method,frames_fused=len(frames),registered_views=len(registered),fusion_images=[scene.names[f['index']] for f in frames],heldout_images=[scene.names[i] for i in heldout],heldout=held,convention=check,roi=[x.tolist() for x in roi],voxel_length=voxel,full=mesh_statistics(full),light=mesh_statistics(light),timing_seconds=timing,peak_rss_mb=float(memory))
    for package in ['open3d','torch','torchvision','transformers','trimesh','scikit-image']:
        try:summary['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    write_json(out/'summary.json',summary);(out/'config.yaml').write_text(yaml.safe_dump(cfg));logging.info('Mesh complete: %s',summary['light']);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--images',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/mesh_default.yaml');p.add_argument('--backend',choices=['mono','sgbm','vggt']);p.add_argument('--cache-only',action='store_true');p.add_argument('--heldout',type=int)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);logging.basicConfig(level=logging.INFO,handlers=[logging.StreamHandler(),logging.FileHandler(a.out/'run.log')],format='%(asctime)s %(message)s')
    cfg=merge(yaml.safe_load((ROOT/'configs/mesh_default.yaml').read_text()),yaml.safe_load(a.config.read_text()) or {})
    if a.heldout is not None:
        if a.heldout<0:p.error('--heldout must be nonnegative')
        cfg['heldout']['num_views']=a.heldout
    try:run(a.scene,a.out,cfg,a.images,a.backend,a.cache_only)
    except Exception as e:
        write_json(a.out/'status.json',dict(status='failed',error=str(e)));raise
    write_json(a.out/'status.json',dict(status='complete'))

if __name__=='__main__':main()
