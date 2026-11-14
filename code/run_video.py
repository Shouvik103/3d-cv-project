"""Select video keyframes automatically and reconstruct a sparse colored cloud."""
import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml

from data.video_frames import extract_video
from run_sfm import run_sfm
from runtime import manifest, write_json

ROOT = Path(__file__).resolve().parents[1]


def run_video(video, output, video_config, sfm_config, *, calibration=None,
              focal_px=None, extract_only=False, preview=True):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new --out directory')
    output.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(output/'run.log')
    log_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(log_handler)
    started = time.perf_counter()
    status = dict(manifest(dict(video=video_config, sfm=sfm_config)), status='extracting', video_name=video.name)
    write_json(output/'status.json', status)
    try:
        report = extract_video(video, output/'input', video_config, calibration, focal_px)
        status['selected_frames'] = len(report['selected_frames'])
        status['selection_report'] = 'input/frames.json'
        if extract_only:
            status['status'] = 'extracted'
        else:
            status['status'] = 'reconstructing'
            write_json(output/'status.json', status)
            reconstruction = run_sfm(output/'input/images', output/'reconstruction', sfm_config)
            count = len(reconstruction.registered_ids())
            status.update(registered=count, points=len(reconstruction.points),
                          registration_rate=count/status['selected_frames'],
                          status='completed' if count == status['selected_frames'] else 'partial',
                          pointcloud='reconstruction/points.ply', poses='reconstruction/poses.json')
            if preview:
                subprocess.run([sys.executable, str(ROOT/'code/viz/render_orbit.py'), '--reconstruction',
                                str(output/'reconstruction'), '--out', str(output/'preview.png')], check=True)
                status['preview'] = 'preview.png'
        status['runtime_seconds'] = time.perf_counter()-started
        write_json(output/'status.json', status)
        return status
    except Exception as exc:
        status.update(status='failed', error=str(exc), runtime_seconds=time.perf_counter()-started)
        write_json(output/'status.json', status)
        raise
    finally:
        root_logger.removeHandler(log_handler)
        log_handler.close()
        root_logger.setLevel(previous_level)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sfm_default.yaml', help='SfM configuration')
    parser.add_argument('--video-config', type=Path, default=ROOT/'configs/video_default.yaml')
    parser.add_argument('--max-frames', type=int, help='Maximum reconstruction views; default 40')
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--calibration', type=Path, help='Checkerboard calibration matching video resolution')
    calibration.add_argument('--focal-px', type=float, help='Focal length in pixels at original video resolution')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    try:
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError('Output directory is not empty; choose a new --out directory')
        args.out.mkdir(parents=True, exist_ok=True)
        video_cfg = yaml.safe_load(args.video_config.read_text())
        sfm_cfg = yaml.safe_load(args.config.read_text())
        if args.max_frames is not None:
            video_cfg['max_frames'] = args.max_frames
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        status = run_video(args.video, args.out, video_cfg, sfm_cfg, calibration=args.calibration,
                           focal_px=args.focal_px, extract_only=args.extract_only, preview=not args.no_preview)
        print(json.dumps(status, indent=2))
        return 2 if status['status'] == 'partial' else 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        logging.error('%s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
