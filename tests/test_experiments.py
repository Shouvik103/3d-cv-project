"""Guard matched-input benchmark provenance and reference interpretation."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from experiments.protocol import select_uniform,prepare_scene


def test_uniform_benchmark_subset_is_deterministic():
    assert select_uniform(['g','b','a','c','f','d','e'],3)==['a','d','g']
    with pytest.raises(ValueError):select_uniform(['a','b'],2)


def test_frozen_inputs_preserve_calibration_and_detect_changes(tmp_path,monkeypatch):
    import experiments.protocol as protocol
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    folder=tmp_path/'raw';folder.mkdir()
    names=['a.jpg','b.jpg','c.jpg']
    for i,name in enumerate(names):Image.new('RGB',(64,48),(i*30,80,100)).save(folder/name)
    K=[[80,0,32],[0,80,24],[0,0,1]]
    records={n:dict(K=K,image_size=[64,48]) for n in names}
    (tmp_path/'intrinsics.yaml').write_text(yaml.safe_dump(dict(images=records)))
    reference=dict(image_names=names,registered=names,K={n:K for n in names},
                   R={n:np.eye(3).tolist() for n in names},t={n:[i,i*i,0] for i,n in enumerate(names)})
    (tmp_path/'reference.json').write_text(json.dumps(reference))
    scene=dict(images='raw',intrinsics='intrinsics.yaml',reference='reference.json',reference_kind='colmap',views=3,notes='test-only pseudo-reference')
    config=dict(image_max_side=32,seed=0)
    out=tmp_path/'prepared'
    first=prepare_scene('test',scene,config,out)
    assert prepare_scene('test',scene,config,out)['prepared_sha256']==first['prepared_sha256']
    actual=yaml.safe_load((out/'intrinsics.yaml').read_text())['images']['a.jpg']
    assert actual['image_size']==[32,24] and np.allclose(actual['K'],[[40,0,16],[0,40,12],[0,0,1]])
    assert first['reference_kind']=='colmap'
    assert json.loads((tmp_path/'reference.json').read_text())==reference
    Image.new('RGB',(64,48),'red').save(folder/'a.jpg')
    with pytest.raises(ValueError,match='Frozen inputs changed'):prepare_scene('test',scene,config,out)


def test_benchmark_evaluation_labels_pseudo_reference(tmp_path):
    from experiments.run_e1 import evaluate
    scene=tmp_path/'scene';scene.mkdir();out=tmp_path/'method';out.mkdir()
    names=['a.jpg','b.jpg','c.jpg','d.jpg'];centers=[[0,0,0],[1,0,0],[0,1,0],[1,1,1]]
    poses=dict(image_names=names,registered=names,K={n:np.eye(3).tolist() for n in names},
               R={n:np.eye(3).tolist() for n in names},t={n:(-np.array(c,dtype=float)).tolist() for n,c in zip(names,centers)})
    (scene/'reference_poses.json').write_text(json.dumps(poses));(scene/'inputs.json').write_text('{}')
    (out/'poses.json').write_text(json.dumps(poses))
    result=evaluate(scene,out,'colmap',dict(method='ours'))
    assert result['reference_kind']=='colmap'
    assert result['metrics']['ate_percent_extent']<1e-10
    assert result['estimated_registered']==4 and result['total_images']==4


def test_two_camera_result_keeps_relative_accuracy_without_fabricating_ate():
    from eval.metrics import evaluate_poses
    names=['a','b','c','d'];C=[[0,0,0],[1,0,0],[0,1,0],[1,1,1]]
    reference=dict(image_names=names,registered=names,R={n:np.eye(3).tolist() for n in names},
                   t={n:(-np.array(c)).tolist() for n,c in zip(names,C)})
    estimate=dict(reference,registered=names[:2])
    with pytest.raises(ValueError):evaluate_poses(estimate,reference)
    result=evaluate_poses(estimate,reference,allow_partial=True)
    assert result['alignment'] is None and result['ate_rmse'] is None
    assert result['registration_rate']==.5
    assert abs(result['auc']['5']-1/6)<1e-10
    assert result['median_rotation_deg']==0 and result['median_translation_deg']==0


def test_ablation_variants_do_not_change_unrelated_factors():
    from copy import deepcopy
    from experiments.run_ablation import variants
    base=dict(ba=dict(mode='local_global',loss='huber'),features=dict(detector='sift',rootsift=True,n_features=4000),
              matching=dict(ratio=.85,ransac_thresh_px=1.5),seed=0)
    original=deepcopy(base)
    for label,config in variants('E2',base).items():
        expected=deepcopy(base);expected['ba']['mode']=label
        assert config==expected
    for label,config in variants('E4',base).items():
        assert config['matching']==base['matching'] and config['ba']==base['ba']
        assert config['features']['rootsift']==(label=='rootsift')
        assert config['features']['detector']==label
    for label,config in variants('E6',base).items():
        changed=[key for key in base['matching'] if config['matching'][key]!=base['matching'][key]]
        assert len(changed)==(0 if label=='baseline' else 1)
    assert base==original


def test_strecha_camera_conversion_matches_world_to_camera_projection(tmp_path):
    from scipy.spatial.transform import Rotation
    from data.download_strecha import read_camera,reference_poses
    root=tmp_path/'fountain_dense/urd';root.mkdir(parents=True)
    K=np.array([[900,0,320],[0,910,240],[0,0,1.]])
    R=Rotation.from_euler('xyz',[.4,-.2,.1]).as_matrix();C=np.array([3.,-4.,2.])
    values=np.r_[K.ravel(),np.zeros(3),R.T.ravel(),C,[640,480]]
    (root/'0000.png.camera').write_text(' '.join(map(str,values)))
    np.savetxt(root/'0000.png.P',K@np.column_stack([R,-R@C]))
    _,converted,t,size=read_camera(root/'0000.png.camera')
    assert size==[640,480] and np.allclose(converted,R) and np.allclose(t,-R@C)
    poses,checks=reference_poses(tmp_path,['0000.jpg'])
    assert checks['0000.jpg']<1e-12 and np.allclose(poses['R']['0000.jpg'],R)
    np.savetxt(root/'0000.png.P',K@np.column_stack([R,C]))
    with pytest.raises(ValueError,match='convention mismatch'): reference_poses(tmp_path,['0000.jpg'])


def test_large_mesh_surface_sampling_is_area_weighted_and_repeatable():
    import open3d as o3d
    from mesh.mesh_eval import sample_surface
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]],float)
    # Cross the 100k-face chunk boundary. Right triangle has four times the area.
    faces=np.tile([[0,1,2],[3,4,5]],(50001,1))
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),o3d.utility.Vector3iVector(faces))
    samples=sample_surface(mesh,10000,3)
    assert np.array_equal(samples,sample_surface(mesh,10000,3))
    assert abs((samples[:,0]>5).mean()-.8)<.02
    assert np.all(samples[:,2]==0)
