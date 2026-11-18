"""Isolated reconstruction process used by the local upload interface."""
from pathlib import Path
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from prepare_data import prepare
from run_sfm import run_sfm
from web.geometry import export_geometry
from web.settings import mesh_quality
from web.mesh_input import prepare_mesh_scene

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
        # An optional known focal length is measured at each original image size.
        if metadata.get('focal_px'):
            from PIL import Image, ImageOps
            path = job / 'input/intrinsics.yaml'
            intrinsics = yaml.safe_load(path.read_text())
            for upload in metadata['uploads']:
                with Image.open(job / 'uploads' / upload['stored']) as raw:
                    width, height = ImageOps.exif_transpose(raw).size
                name = Path(upload['stored']).stem + '.jpg'
                record = intrinsics['images'][name]
                scale = record['image_size'][0] / width
                record['K'][0][0] = record['K'][1][1] = metadata['focal_px'] * scale
                record['source'] = 'user_focal_px_at_original_resolution'
            path.write_text(yaml.safe_dump(intrinsics))
        state('reconstructing', 'Matching views and estimating camera poses', 15)
        cfg = yaml.safe_load((ROOT / 'configs/sfm_relaxed.yaml').read_text())
        cfg['export']['save_growth_snapshots'] = False
        run_sfm(job / 'input/images', job / 'reconstruction', cfg, progress=work_progress)
        summary = json.loads((job / 'reconstruction/summary.json').read_text())
        metrics.update(images=summary['total_images'], registered=summary['registered'],
                       sparse_points=summary['points'], reprojection_px=summary['median_reprojection_px'])
        artifacts.update(sparse_cloud='reconstruction/points.ply', poses='reconstruction/poses.json',
                         reconstruction_summary='reconstruction/summary.json')
        if summary['registered'] < summary['total_images']:
            warnings.append(f"Only {summary['registered']} of {summary['total_images']} images registered. More overlapping views may help.")
        state('point_cloud', 'Preparing the interactive point cloud', 55)
        export_geometry(job / artifacts['sparse_cloud'], job / 'cloud.json', job / artifacts['poses'])
        artifacts['cloud_view'] = 'cloud.json'
        state('point_cloud', 'Point cloud ready', 60)
        if metadata['make_mesh']:
            try:
                quality = mesh_quality(metadata)
                metrics.update(mesh_quality=quality['label'], mesh_depth_side=quality['depth_side'],
                               mesh_voxel_div=quality['voxel_div'])
                state('meshing', f"Preparing {quality['label'].lower()} quality mesh images", 62)
                mesh_scene = prepare_mesh_scene(job, quality['depth_side'])
                state('meshing', f"Estimating stereo depth · up to {quality['depth_side']} px", 65)
                from mesh.run_mesh import run, merge
                cfg = merge(yaml.safe_load((ROOT / 'configs/mesh_default.yaml').read_text()),
                            yaml.safe_load((ROOT / 'configs/mesh_object.yaml').read_text()))
                # This is a user reconstruction, rather than an evaluation run:
                # every registered view is available for stereo fusion.
                cfg['heldout']['num_views'] = 0
                cfg['sgbm']['input_long_side'] = quality['depth_side']
                cfg['tsdf']['voxel_div'] = quality['voxel_div']
                mesh = run(mesh_scene, job / 'mesh', cfg, backend='sgbm', progress=work_progress)
                artifacts.update(mesh_ply='mesh/mesh_light.ply', mesh_obj='mesh/mesh_light.obj',
                                 mesh_glb='mesh/mesh_light.glb', full_mesh='mesh/mesh_full.ply',
                                 dense_cloud='mesh/dense.ply', mesh_summary='mesh/summary.json')
                artifacts = {k: v for k, v in artifacts.items() if (job / v).is_file()}
                metrics.update(triangles=mesh['light']['faces'], vertices=mesh['light']['vertices'],
                               watertight=mesh['light']['watertight'])
                state('exporting', 'Preparing interactive mesh and dense cloud', 92)
                export_geometry(job / artifacts['mesh_ply'], job / 'mesh.json', job / artifacts['poses'], 'mesh')
                artifacts['mesh_view'] = 'mesh.json'
                state('exporting', 'Preparing the dense point cloud', 96)
                export_geometry(job / artifacts['dense_cloud'], job / 'cloud.json', job / artifacts['poses'])
                metrics['dense_points'] = json.loads((job / 'cloud.json').read_text())['total_points']
            except InterruptedError:
                raise
            except Exception as exc:
                logging.exception('Mesh stage failed; retaining the point cloud')
                warnings.append(f'Mesh could not be completed: {exc}. The point cloud is still available.')
        status = 'partial' if warnings else 'completed'
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
