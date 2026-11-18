"""Isolated reconstruction process used by the local upload interface."""
from pathlib import Path
import argparse
import json
import logging
import os
import sys
import time
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from prepare_data import prepare
from run_sfm import run_sfm
from web.geometry import export_geometry
from web.settings import mesh_quality
from web.mesh_input import prepare_mesh_scene,upload_masks
from web.settings import METHOD_NAMES

ROOT = Path(__file__).resolve().parents[2]


def write_state(job, **values):
    path = job / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(values, updated_at=time.time())
    tmp = path.with_name(f'state-{os.getpid()}.tmp')
    tmp.write_text(json.dumps(state, indent=2, allow_nan=False))
    tmp.replace(path)
    return state


def run_job(job):
    job = Path(job)
    metadata = json.loads((job / 'job.json').read_text())
    started = time.time()
    artifacts, warnings, metrics = {}, [], {}
    method=metadata.get('method','ours')
    metrics.update(method=method,quality_profile=metadata.get('quality_profile','standard'))
    import resource

    def state(stage, message, progress, **extra):
        if (job / 'cancelled').exists():
            raise InterruptedError('Cancelled by user')
        extra.setdefault('progress_units', None)
        return write_state(job, status='running', stage=stage, message=message,
                           progress=progress, artifacts=artifacts, metrics=metrics,
                           warnings=warnings, **extra)

    last_progress_update = 0
    ranges = dict(features=(15,22),matching=(22,42),initialization=(42,45),
                  registration=(45,52),optimization=(52,54),sfm_export=(54,55),
                  depth=(65,85),filtering=(85,87),fusion=(87,89),
                  cleanup=(89,91),mesh_preview=(91,92))

    def work_progress(phase, done, total, message):
        nonlocal last_progress_update
        now = time.monotonic()
        if now - last_progress_update < .4 and done not in (0,total):
            return
        last_progress_update = now
        lower, upper = ranges[phase]
        value = lower + (upper-lower)*done/max(total,1)
        stage = 'reconstructing' if lower < 55 else 'meshing'
        state(stage, message, round(value,1), progress_units=dict(phase=phase,done=done,total=total))

    try:
        state('preparing', 'Preparing images and camera intrinsics', 5)
        prepare(job / 'uploads', job / 'input/images', metadata['max_side'])
        masks=upload_masks(job,job/'input/images',job/'input_masks')
        # User-supplied intrinsics are measured at the original upload resolution.
        if metadata.get('focal_px') or metadata.get('camera_matrix'):
            import numpy as np
            from PIL import Image, ImageOps
            path = job / 'input/intrinsics.yaml'
            intrinsics = yaml.safe_load(path.read_text())
            for upload in metadata['uploads']:
                with Image.open(job / 'uploads' / upload['stored']) as raw:
                    width, height = ImageOps.exif_transpose(raw).size
                name = Path(upload['stored']).stem + '.jpg'
                record = intrinsics['images'][name]
                scale = record['image_size'][0] / width
                if metadata.get('camera_matrix'):
                    K=np.array(metadata['camera_matrix'],float)
                    if K[0,2]>=width or K[1,2]>=height:raise ValueError('Camera principal point must be inside the original image')
                    K[0]*=record['image_size'][0]/width;K[1]*=record['image_size'][1]/height
                    record['K']=K.tolist();record['source']='user_camera_matrix_at_original_resolution'
                else:
                    record['K'][0][0] = record['K'][1][1] = metadata['focal_px'] * scale
                    record['source'] = 'user_focal_px_at_original_resolution'
            path.write_text(yaml.safe_dump(intrinsics))
        object_quality=metadata.get('quality_profile')=='object'
        if object_quality:
            import shutil
            from web.foreground import prepare_foreground
            masks=prepare_foreground(job)
            shutil.copytree(masks,job/'input/masks',dirs_exist_ok=True)
            artifacts['foreground_report']='foreground.json'
        state('reconstructing',METHOD_NAMES[method]+': estimating cameras and geometry',15)
        if method=='ours':
            cfg = yaml.safe_load((ROOT / 'configs/sfm_relaxed.yaml').read_text())
            cfg['export']['save_growth_snapshots'] = False
            cfg['matching']['ratio']=metadata.get('matching_ratio',.85)
            if object_quality:
                cfg['conflict_aware_tracks']=True
                cfg['triangulation']['robust']=True
                cfg['filter']['prune_observations']=True
                cfg['matching']['bounded_pairs']=True
                cfg['features']['n_features']=6000
                cfg['init']['top_k_pairs']=40
                cfg['init']['max_hf_ratio']=.9
                cfg['ba'].update(max_points_global=2500,max_nfev=40,max_nfev_local=20,global_every=10)
                cfg['ba']['refine_focal']=not bool(metadata.get('camera_matrix') or metadata.get('focal_px'))
            run_sfm(job / 'input/images', job / 'reconstruction', cfg, progress=work_progress)
        elif method=='colmap':
            command=[sys.executable,str(ROOT/'code/web/colmap_job.py'),'--images',str(job/'input/images'),'--out',str(job/'reconstruction')]
            if object_quality:command+=['--object-quality']
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
            for line in process.stdout:
                logging.info('%s',line.rstrip())
                if line.startswith('WEB_PROGRESS '):work_progress(*json.loads(line.removeprefix('WEB_PROGRESS ')))
            if process.wait()!=0:raise RuntimeError('COLMAP failed; inspect the method run log.')
        elif method=='vggt':
            python=ROOT/'outputs/vggt_env/bin/python'
            if not python.exists():raise RuntimeError('VGGT runtime is not installed. Run .venv/bin/python code/experiments/setup_vggt.py first.')
            command=[str(python),str(ROOT/'code/web/vggt_job.py'),'--images',str(job/'input/images'),'--out',str(job/'reconstruction')]
            if masks:command+=['--masks',str(masks)]
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                     env=dict(os.environ,PYTHONUNBUFFERED='1',PYTORCH_ENABLE_MPS_FALLBACK='1'))
            window_index,window_count=0,1
            for line in process.stdout:
                logging.info('%s',line.rstrip())
                if 'VGGT window ' in line:
                    import re
                    found=re.search(r'VGGT window (\d+)/(\d+)',line)
                    if found:window_index,window_count=int(found[1])-1,int(found[2])
                if 'aggregator: input shape' in line:state('reconstructing',f'VGGT: aggregating window {window_index+1}/{window_count}',round(15+35*(window_index+.2)/window_count,1))
                elif 'prediction heads (' in line:state('reconstructing',f'VGGT: predicting window {window_index+1}/{window_count}',round(15+35*(window_index+.7)/window_count,1))
            if process.wait()!=0:raise RuntimeError('VGGT inference failed; inspect the run log for the model/runtime error.')
            artifacts.update(raw_predictions='reconstruction/raw.npz',vggt_metadata='reconstruction/raw.json')
            warnings.append('VGGT cameras are predictions, not feature-verified registrations. The point export is confidence-filtered and capped at 1 million; raw dense arrays are available.')
            if len(metadata['uploads'])>16:
                warnings.append('VGGT uses overlapping batches of at most 16 views with checked shared-camera Sim(3) alignment. This bounds memory but can introduce stitching seams; it is not one joint 60-view prediction.')
                metrics['inference_mode']='Overlapping 16-view batches'
        else:raise ValueError('Unknown reconstruction method')
        summary = json.loads((job / 'reconstruction/summary.json').read_text())
        metrics.update(images=summary['total_images'], registered=summary['registered'],
                       sparse_points=summary['points'], reprojection_px=summary.get('median_reprojection_px',summary.get('mean_reprojection_px')),
                       reprojection_statistic='median' if 'median_reprojection_px' in summary else ('mean' if 'mean_reprojection_px' in summary else None),
                       reconstruction_seconds=summary.get('runtime_seconds'))
        if method=='vggt':metrics['dense_points']=summary['points']
        artifacts.update(sparse_cloud='reconstruction/points.ply', poses='reconstruction/poses.json',
                         reconstruction_summary='reconstruction/summary.json')
        if summary['registered'] < summary['total_images']:
            warnings.append(f"Only {summary['registered']} of {summary['total_images']} images registered. Some views failed camera/track checks; inspect overlap, blur and calibration.")
        state('point_cloud', 'Preparing the interactive point cloud', 55)
        export_geometry(job / artifacts['sparse_cloud'], job / 'cloud.json', job / artifacts['poses'])
        artifacts['cloud_view'] = 'cloud.json'
        state('point_cloud', 'Point cloud ready', 60)
        if metadata['make_mesh']:
            try:
                quality = mesh_quality(metadata)
                depth_backend='vggt' if method=='vggt' else metadata.get('depth_backend','sgbm')
                metrics.update(mesh_quality=quality['label'], mesh_depth_side=518 if method=='vggt' else quality['depth_side'],
                               mesh_voxel_div=quality['voxel_div'])
                state('meshing', f"Preparing {quality['label'].lower()} quality mesh images", 62)
                mesh_scene = job/'reconstruction/raw.npz' if method=='vggt' else prepare_mesh_scene(job, quality['depth_side'])
                state('meshing', f"Estimating {depth_backend.upper()} depth · up to {518 if method=='vggt' else quality['depth_side']} px", 65)
                from mesh.run_mesh import run, merge
                cfg = merge(yaml.safe_load((ROOT / 'configs/mesh_default.yaml').read_text()),
                            yaml.safe_load((ROOT / 'configs/mesh_object.yaml').read_text()))
                # This is a user reconstruction, rather than an evaluation run:
                # every registered view is available for stereo fusion.
                cfg['heldout']['num_views'] = 0
                cfg['sgbm']['input_long_side'] = quality['depth_side']
                cfg['mono']['input_long_side']=quality['depth_side']
                cfg['tsdf']['voxel_div'] = quality['voxel_div']
                if depth_backend=='mono' and metadata.get('comparison_id'):
                    scene_file=mesh_scene/'scene.json';scene_meta=json.loads(scene_file.read_text())
                    scene_meta['depth_cache']=str(job.parent/metadata['comparison_id']/'depth_cache')
                    scene_file.write_text(json.dumps(scene_meta))
                metrics['depth_backend']=depth_backend
                mesh = run(mesh_scene, job / 'mesh', cfg, backend=depth_backend, progress=work_progress)
                artifacts.update(mesh_ply='mesh/mesh_light.ply', mesh_obj='mesh/mesh_light.obj',
                                 mesh_glb='mesh/mesh_light.glb', full_mesh='mesh/mesh_full.ply',
                                 dense_cloud='mesh/dense.ply', mesh_summary='mesh/summary.json')
                artifacts = {k: v for k, v in artifacts.items() if (job / v).is_file()}
                metrics.update(triangles=mesh['light']['faces'], vertices=mesh['light']['vertices'],
                               watertight=mesh['light']['watertight'],mesh_seconds=mesh['timing_seconds']['total'],
                               components=mesh['full']['components'],boundary_edges=mesh['full']['boundary_edges'])
                state('exporting', 'Preparing interactive mesh and dense cloud', 92)
                export_geometry(job / artifacts['mesh_ply'], job / 'mesh.json', job / artifacts['poses'], 'mesh')
                artifacts['mesh_view'] = 'mesh.json'
                state('exporting', 'Preparing the dense point cloud', 96)
                if method!='vggt':
                    export_geometry(job / artifacts['dense_cloud'], job / 'cloud.json', job / artifacts['poses'])
                    metrics['dense_points'] = json.loads((job / 'cloud.json').read_text())['total_points']
            except InterruptedError:
                raise
            except Exception as exc:
                logging.exception('Mesh stage failed; retaining the point cloud')
                warnings.append(f'Mesh could not be completed: {exc}. The point cloud is still available.')
        metrics['peak_rss_mb']=max(float(summary.get('peak_rss_mb',0)),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024))
        # Informational VGGT notes do not turn successful results into failures.
        status = 'partial' if summary['registered']<summary['total_images'] or (metadata['make_mesh'] and 'mesh_view' not in artifacts) else 'completed'
        write_state(job, status=status, stage='finished', message='Reconstruction ready', progress=100,
                    artifacts=artifacts, metrics=metrics, warnings=warnings, elapsed_seconds=time.time() - started)
    except InterruptedError:
        write_state(job, status='cancelled', stage='finished', message='Run cancelled', artifacts=artifacts)
    except Exception as exc:
        logging.exception('Reconstruction failed')
        write_state(job, status='failed', stage='finished', message=str(exc), artifacts=artifacts,
                    metrics=metrics, warnings=warnings, elapsed_seconds=time.time() - started)
        return 1
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    raise SystemExit(run_job(args.job))
