from pathlib import Path
import json
import subprocess
import sys
from eval.metrics import evaluate_poses
from sfm.io_utils import load_poses_json

ROOT = Path(__file__).resolve().parents[1]


def test_sample_cli(tmp_path):
    subprocess.run([sys.executable, str(ROOT/'code/run_sfm.py'), '--images', str(ROOT/'sample_data/scene_small/images'),
                    '--out', str(tmp_path/'ours'), '--config', str(ROOT/'configs/sfm_default.yaml')], check=True, timeout=240, cwd=tmp_path)
    summary = json.loads((tmp_path/'ours/summary.json').read_text())
    assert summary['registered'] == 8
    assert summary['points'] > 100
    assert summary['median_reprojection_px'] < 1.5
    metrics = evaluate_poses(load_poses_json(tmp_path/'ours/poses.json'), load_poses_json(ROOT/'sample_data/scene_small/gt_poses.json'))
    assert metrics['ate_percent_extent'] < 1.


def test_registration_retries_after_failed_candidate(tmp_path, monkeypatch):
    import cv2
    import yaml
    from run_sfm import run_sfm
    original = cv2.solvePnPRansac
    calls = []
    def reject_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1: return False, None, None, None
        return original(*args, **kwargs)
    monkeypatch.setattr(cv2, 'solvePnPRansac', reject_once)
    config = yaml.safe_load((ROOT/'configs/sfm_default.yaml').read_text())
    updates = []
    reconstruction = run_sfm(ROOT/'sample_data/scene_small/images', tmp_path/'retry', config,
                             progress=lambda phase, done, total, message: updates.append((phase, done, total)))
    assert len(reconstruction.registered_ids()) == 8
    assert len(calls) >= 7
    matching = [(done,total) for phase,done,total in updates if phase == 'matching']
    assert matching[0] == (0,28) and matching[-1] == (28,28)
    assert [done for done,total in matching] == list(range(29))
    assert updates[-1] == ('sfm_export',1,1)
