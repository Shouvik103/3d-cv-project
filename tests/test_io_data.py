from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from prepare_data import prepare
from data.download_middlebury import parse_cameras
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




def test_orb_remains_binary_when_rootsift_flag_is_set():
    from sfm.features import extract_features
    image = np.random.default_rng(0).integers(0, 256, (200, 200), dtype=np.uint8)
    _, descriptors = extract_features(image, 'orb', rootsift=True)
    assert descriptors.dtype == np.uint8


def test_prepare_reads_nested_phone_focal_length(tmp_path):
    raw = tmp_path/'raw'; raw.mkdir()
    image = Image.new('RGB', (1200, 900))
    exif = image.getexif(); exif[34665] = {41989: 24}
    image.save(raw/'phone.jpg', exif=exif)
    result = prepare(raw, tmp_path/'prepared/images', max_side=600)
    record = result['images']['phone.jpg']
    assert record['source'] == 'exif_35mm'
    assert np.isclose(record['K'][0][0], 400.)




def test_ply_roundtrip_including_empty_cloud(tmp_path):
    points=np.array([[1.,2.,3.],[4.,5.,6.]])
    colors=np.array([[20,80,120],[50,100,160]],dtype='u1')
    save_ply(points,colors,tmp_path/'color.ply')
    actual,rgb=load_ply(tmp_path/'color.ply')
    assert np.allclose(actual,points) and np.array_equal(rgb,colors)
    save_ply(np.empty((0,3)),None,tmp_path/'empty.ply')
    assert load_ply(tmp_path/'empty.ply')[0].shape==(0,3)
