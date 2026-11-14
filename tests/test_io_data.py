from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from prepare_data import prepare
from data.download_middlebury import parse_cameras
from baselines.convert_to_common import export_vggt
from sfm.io_utils import load_ply, save_ply, load_poses_json


def test_prepare_preserves_calibration(tmp_path):
    images = tmp_path/'raw'; images.mkdir()
    Image.new('RGB', (1200, 800)).save(images/'a.png')
    K = [[1000., 0, 600], [0, 1000, 400], [0, 0, 1]]
    intrinsics = tmp_path/'input.yaml'; intrinsics.write_text(yaml.safe_dump(dict(images={'a.png': dict(K=K, image_size=[1200, 800])})))
    result = prepare(images, tmp_path/'prepared'/'images', 600, intrinsics)
    actual = np.array(result['images']['a.jpg']['K'])
    assert np.allclose(actual, [[500, 0, 300], [0, 500, 200], [0, 0, 1]])
    assert result['images']['a.jpg']['image_size'] == [600, 400]


def test_middlebury_parser(tmp_path):
    K = np.diag([700, 700, 1]); R = np.eye(3); t = np.array([1, 2, 3])
    path = tmp_path/'params.txt'
    path.write_text('1\nimage.png ' + ' '.join(map(str, np.r_[K.ravel(), R.ravel(), t])) + '\n')
    a, b, c = parse_cameras(path)['image.png']
    assert np.array_equal(a, K) and np.array_equal(b, R) and np.array_equal(c, t)


def test_vggt_roundtrip_and_ply(tmp_path):
    names = np.array(['a.png', 'b.png', 'c.png'])
    E = np.tile(np.c_[np.eye(3), np.zeros(3)], (3, 1, 1)); E[:, :, 3] = np.eye(3)
    K = np.tile(np.eye(3), (3, 1, 1))
    path = tmp_path/'vggt.npz'
    np.savez(path, image_names=names, extrinsic=E, intrinsic=K, world_points=np.ones((3, 2, 2, 3)), world_points_conf=np.ones((3, 2, 2)))
    out = tmp_path/'out'; export_vggt(path, out)
    poses = load_poses_json(out/'poses.json')
    assert poses['registered'] == names.tolist()
    assert np.array_equal(poses['t']['b.png'], E[1, :, 3])
    assert load_ply(out/'points.ply')[0].shape == (12, 3)
    save_ply(np.empty((0, 3)), None, tmp_path/'empty.ply')
    assert load_ply(tmp_path/'empty.ply')[0].shape == (0, 3)


def test_orb_remains_binary_when_rootsift_flag_is_set():
    from sfm.features import extract_features
    image = np.random.default_rng(0).integers(0, 256, (200, 200), dtype=np.uint8)
    _, descriptors = extract_features(image, 'orb', rootsift=True)
    assert descriptors.dtype == np.uint8
