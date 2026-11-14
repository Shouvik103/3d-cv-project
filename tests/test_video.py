import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
import yaml

from data.video_frames import extract_video, validate_settings
from eval.metrics import evaluate_poses
from run_video import run_video
from sfm.io_utils import load_poses_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def video_config():
    return yaml.safe_load((ROOT/'configs/video_default.yaml').read_text())


@pytest.fixture
def video_scene(tmp_path):
    path = tmp_path/'uneven_motion.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 12., (800, 600))
    assert writer.isOpened()
    source_views, blurred = [], set()
    try:
        for view, hold in enumerate([12, 24, 10, 28, 12, 20, 16, 10]):
            image = cv2.imread(str(ROOT/f'sample_data/scene_small/images/view_{view:02d}.png'))
            if view:
                blur = cv2.GaussianBlur(image, (31, 31), 8)
                for _ in range(3):
                    blurred.add(len(source_views)); source_views.append(view); writer.write(blur)
            for _ in range(hold):
                source_views.append(view); writer.write(image)
        # A separate static shot must not be spliced into the moving object's views.
        unrelated = np.random.default_rng(9).integers(0, 256, (600, 800, 3), dtype=np.uint8)
        for _ in range(18):
            source_views.append(-1); writer.write(unrelated)
    finally:
        writer.release()
    return path, source_views, blurred


def test_content_based_selection(video_scene, video_config, tmp_path):
    video, views, blurred = video_scene
    video_config['max_frames'] = 6
    video_config['max_side'] = 400
    report = extract_video(video, tmp_path/'frames', video_config, focal_px=700.)
    selected = report['selected_frames']
    indices = [record['frame_index'] for record in selected]
    assert 5 <= len(indices) <= 6
    assert len(set(views[index] for index in indices)) == len(indices)
    assert not set(indices) & blurred
    assert all(views[index] >= 0 for index in indices)
    assert report['selection']['continuous_shots'] >= 2
    assert len(set(np.diff(indices))) > 1  # Content, rather than fixed time offsets.
    for record in selected:
        assert abs(record['timestamp_seconds']-record['frame_index']/12) < .1
    calibration = yaml.safe_load((tmp_path/'frames/intrinsics.yaml').read_text())
    for record in calibration['images'].values():
        assert record['image_size'] == [400, 300]
        assert np.allclose(record['K'], [[350, 0, 200], [0, 350, 150], [0, 0, 1]])
    assert (tmp_path/'frames/selected_frames.png').exists()
    assert not list((tmp_path/'frames').glob('frame-scan-*'))


def test_video_cli_reconstructs(video_scene, tmp_path):
    video, views, _ = video_scene
    out = tmp_path/'reconstruction'
    subprocess.run([sys.executable, str(ROOT/'code/run_video.py'), '--video', str(video),
                    '--out', str(out), '--focal-px', '700'], cwd=tmp_path, check=True, timeout=240)
    status = json.loads((out/'status.json').read_text())
    assert status['status'] == 'completed'
    assert status['registered'] >= 5
    assert status['points'] > 100
    assert (out/'preview.png').exists()
    assert (out/'run.log').stat().st_size > 0
    chosen = json.loads((out/'input/frames.json').read_text())['selected_frames']
    truth = load_poses_json(ROOT/'sample_data/scene_small/gt_poses.json')
    reference = dict(image_names=[], registered=[], R={}, t={})
    for record in chosen:
        name = record['image_name']; source = f"view_{views[record['frame_index']]:02d}.png"
        reference['image_names'].append(name); reference['registered'].append(name)
        reference['R'][name], reference['t'][name] = truth['R'][source], truth['t'][source]
    metrics = evaluate_poses(load_poses_json(out/'reconstruction/poses.json'), reference)
    assert metrics['ate_percent_extent'] < 1.


def test_static_video_fails_with_diagnostics(video_config, tmp_path):
    path = tmp_path/'static.avi'
    image = cv2.imread(str(ROOT/'sample_data/scene_small/images/view_00.png'))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 12., (800, 600))
    assert writer.isOpened()
    for _ in range(72): writer.write(image)
    writer.release()
    sfm = yaml.safe_load((ROOT/'configs/sfm_default.yaml').read_text())
    with pytest.raises(ValueError, match='Too little viewpoint change'):
        run_video(path, tmp_path/'out', video_config, sfm)
    assert json.loads((tmp_path/'out/status.json').read_text())['status'] == 'failed'
    assert not (tmp_path/'out/reconstruction/points.ply').exists()


def test_existing_output_is_preserved(video_config, tmp_path):
    out = tmp_path/'out'; out.mkdir()
    sentinel = out/'keep.txt'; sentinel.write_text('keep')
    with pytest.raises(ValueError, match='not empty'):
        run_video(tmp_path/'unused.mp4', out, video_config, {})
    assert sentinel.read_text() == 'keep'


def test_missing_video_is_reported(video_config, tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        run_video(tmp_path/'missing.mp4', tmp_path/'out', video_config, {})
    assert 'does not exist' in json.loads((tmp_path/'out/status.json').read_text())['error']


def test_frame_budget_preserves_overlap(video_scene, video_config, tmp_path):
    video_config.update(min_frames=2, max_frames=2, target_motion_fraction=.004)
    with pytest.raises(ValueError, match='Cannot preserve overlap'):
        extract_video(video_scene[0], tmp_path/'budget', video_config)
    assert not list((tmp_path/'budget').glob('frame-scan-*'))


@pytest.mark.parametrize('key,value', [('max_frames', 1), ('candidate_hz', 0), ('min_tracked_fraction', 2),
                                      ('max_side', -2), ('target_motion_fraction', 0), ('max_frames', 5.5),
                                      ('min_sharpness', float('nan'))])
def test_invalid_settings(video_config, key, value):
    video_config[key] = value
    with pytest.raises(ValueError):
        validate_settings(video_config)
