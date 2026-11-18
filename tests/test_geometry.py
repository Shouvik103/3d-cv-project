import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sfm.geometry import fundamental_8pt, sampson_distance, ransac_fundamental, essential_from_F, cheirality_pose, triangulate_dlt, pnp_dlt
from sfm.bundle_adjust import bundle_adjust
from sfm.tracks import build_tracks
from eval.align import umeyama, transform_points
from eval.metrics import evaluate_poses, pose_auc, cloud_metrics, rotation_error, angle_deg
from synthetic import make_scene


def test_generator_and_geometry():
    K, Rs, ts, X, xy = make_scene()
    assert np.allclose(np.linalg.det(Rs), 1)
    F = fundamental_8pt(xy[0], xy[-1])
    assert max(sampson_distance(F, xy[0], xy[-1])) < 1e-12
    # Exercise the underdetermined, exactly-eight-correspondence SVD.
    F8 = fundamental_8pt(xy[0, :8], xy[-1, :8])
    assert max(sampson_distance(F8, xy[0], xy[-1])) < 1e-10
    R, t = cheirality_pose(essential_from_F(F, K, K), xy[0], xy[-1], K, K)
    expected_R = Rs[-1] @ Rs[0].T
    expected_t = ts[-1] - expected_R @ ts[0]
    assert rotation_error(R @ expected_R.T) < 1e-4
    assert angle_deg(t, expected_t) < 1e-4
    for j in range(10):
        assert np.linalg.norm(triangulate_dlt(xy[:, j], Rs, ts, [K]*len(Rs)) - X[j]) < 1e-8
    R, t = pnp_dlt(X, xy[2], K)
    assert rotation_error(R @ Rs[2].T) < 1e-4
    assert np.linalg.norm(t-ts[2]) < 1e-8


def test_ransac_outliers():
    _, _, _, _, xy = make_scene(n_points=300, noise=.3)
    y = xy[-1].copy(); y[:90] = np.random.default_rng(1).uniform([0, 0], [800, 600], (90, 2))
    _, mask = ransac_fundamental(xy[0], y, iters=4000, seed=8)
    assert mask[90:].mean() >= .9
    assert mask[:90].mean() < .1


def test_ba_multiple_free_cameras():
    K, Rs, ts, X, xy = make_scene(n_points=50, n_cameras=5, noise=.3)
    rng = np.random.default_rng(4)
    init_R = Rs.copy(); init_t = ts.copy()
    init_R[2:] = Rotation.from_rotvec(rng.normal(0, .015, (3, 3))).as_matrix() @ Rs[2:]
    init_t[2:] += rng.normal(0, .03, (3, 3))
    ci = np.repeat(np.arange(5), len(X)); pi = np.tile(np.arange(len(X)), 5)
    result = bundle_adjust(init_R, init_t, X+rng.normal(0, .03, X.shape), ci, pi, xy.reshape(-1, 2), np.tile(K, (len(ci), 1, 1)), max_nfev=100)
    assert result.final_rms < .4
    assert result.final_rms < result.initial_rms/10
    assert np.allclose(result.rotations[0], Rs[0], atol=1e-14)
    def centers(R, t): return -np.einsum('nji,nj->ni', R, t)
    assert np.linalg.norm(centers(result.rotations, result.translations)-centers(Rs, ts)) < np.linalg.norm(centers(init_R, init_t)-centers(Rs, ts))


def test_shared_focal_refinement_recovers_calibration_with_known_cameras():
    K,Rs,ts,X,xy=make_scene(n_points=60,n_cameras=5,noise=.15)
    ci=np.repeat(np.arange(5),len(X));pi=np.tile(np.arange(len(X)),5)
    guessed=K.copy();guessed[0,0]*=1.25;guessed[1,1]*=1.25
    intrinsics=np.tile(guessed,(len(ci),1,1));rng=np.random.default_rng(8)
    initial=X+rng.normal(0,.02,X.shape)
    fixed=bundle_adjust(Rs,ts,initial,ci,pi,xy.reshape(-1,2),intrinsics,fixed_camera_ids=tuple(range(5)),max_nfev=100)
    refined=bundle_adjust(Rs,ts,initial,ci,pi,xy.reshape(-1,2),intrinsics,fixed_camera_ids=tuple(range(5)),max_nfev=150,refine_focal=True)
    assert fixed.focal_scale==1.
    assert abs(refined.focal_scale-.8)<.02
    assert refined.final_rms<.3 and refined.final_rms<fixed.final_rms/3
    assert np.allclose(refined.rotations,Rs) and np.allclose(refined.translations,ts)


def test_alignment_and_metrics():
    rng = np.random.default_rng(2); X = rng.normal(size=(10, 3))
    Q = Rotation.from_rotvec([.2, -.4, .1]).as_matrix(); Y = 2.3*X @ Q.T + [2, 1, -3]
    s, R, t = umeyama(X, Y)
    assert np.allclose(transform_points(X, s, R, t), Y, atol=1e-10)
    with pytest.raises(ValueError): umeyama(np.zeros((3, 3)), np.zeros((3, 3)))
    assert pose_auc([0, 0, np.inf, np.inf], 5) == .5
    assert cloud_metrics(X, X)['fscore'] == 1
    K, Rs, ts, _, _ = make_scene()
    names = [str(i) for i in range(len(Rs))]
    poses = dict(image_names=names, registered=names, R=dict(zip(names, Rs.tolist())), t=dict(zip(names, ts.tolist())))
    assert evaluate_poses(poses, poses)['ate_rmse'] < 1e-10


def test_conflicting_tracks_rejected():
    pairs = {(0, 1): np.array([[0, 0], [1, 1]]), (1, 2): np.array([[0, 0], [1, 0]])}
    assert build_tracks(pairs) == {}


def test_auc_counts_unregistered_and_skips_zero_reference_baselines():
    K, Rs, ts, _, _ = make_scene(n_cameras=4)
    names = ['0', '1', '2', '3']
    reference = dict(image_names=names, registered=names, R=dict(zip(names, Rs.tolist())), t=dict(zip(names, ts.tolist())))
    estimate = dict(reference, registered=names[:3])
    metrics = evaluate_poses(estimate, reference)
    assert abs(metrics['auc']['30'] - .5) < 1e-6
    reference['R']['3'] = reference['R']['0']; reference['t']['3'] = reference['t']['0']
    metrics = evaluate_poses(reference, reference)
    assert metrics['undefined_reference_pairs'] == 1
    assert metrics['auc']['30'] > .99999


def test_ransac_extremely_low_inlier_ratio(monkeypatch):
    import sfm.geometry as geometry
    points = np.random.default_rng(5).normal(size=(1000, 2))
    monkeypatch.setattr(geometry, 'sampson_distance', lambda F, a, b: np.r_[0., np.full(len(a)-1, 100.)])
    with pytest.raises(RuntimeError, match='RANSAC failed'):
        geometry.ransac_fundamental(points, points, iters=2)
