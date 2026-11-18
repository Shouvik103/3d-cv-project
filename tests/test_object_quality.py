"""Object quality invariants: track preservation, matching budget, VGGT gauge."""
import numpy as np
import pytest
from sfm.tracks import build_tracks
from sfm.pair_selection import select_pairs
from web.vggt_batches import windows,camera_alignment,transform,centers
from web.foreground import object_mask
from PIL import Image


def test_conflicting_edge_does_not_destroy_two_valid_tracks():
    pairs={(0,1):np.array([[0,0],[1,1],[2,2]]),(1,2):np.array([[0,0],[1,1]]),(0,2):np.array([[0,1]])}
    tracks=build_tracks(pairs,conflict_aware=True)
    assert {frozenset(t.items()) for t in tracks.values()} >= {frozenset({0:0,1:0,2:0}.items()),frozenset({0:1,1:1,2:1}.items())}
    assert len(build_tracks(pairs))<len(tracks)


def test_matching_retains_local_coverage_and_bounds_work():
    images=[np.full((20,20,3),i*4,dtype='u1') for i in range(60)]
    pairs=select_pairs(images)
    assert len(pairs)<900
    assert (0,59) in pairs and (12,18) in pairs
    assert len(pairs)==len(set(pairs))
    assert all(0<=a<b<60 for a,b in pairs)


def test_vggt_windows_cover_sixty_without_sixty_joint_tokens():
    batches=windows(60)
    assert set.union(*(set(b) for b in batches))==set(range(60))
    assert max(map(len,batches))<=16
    for i in range(1,len(batches)):assert len(set(batches[i])&set.union(*(set(b) for b in batches[:i])))>=7


def test_vggt_camera_depth_and_point_transform_share_gauge():
    rng=np.random.default_rng(4);c=rng.normal(size=(12,3));ext=np.tile(np.c_[np.eye(3),np.zeros(3)],(12,1,1));ext[:,:,3]=-c
    angle=.7;Q=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    s=2.3;shift=np.array([1.,-2.,.5]);points=rng.normal(size=(12,2,2,3));depth=np.ones((12,2,2,1))
    transformed=transform(dict(extrinsic=ext,depth=depth,world_points=points),s,Q,shift)
    fitted,R,t,quality=camera_alignment(ext,transformed['extrinsic'])
    assert np.allclose([fitted],s) and np.allclose(R,Q) and np.allclose(t,shift)
    assert np.allclose(centers(transformed['extrinsic']),s*c@Q.T+shift)
    original=np.einsum('nij,nhwj->nhwi',ext[:,:,:3],points)+ext[:,:,3][:,None,None]
    changed=np.einsum('nij,nhwj->nhwi',transformed['extrinsic'][:,:,:3],transformed['world_points'])+transformed['extrinsic'][:,:,3][:,None,None]
    assert np.allclose(changed,s*original,atol=2e-6)
    assert np.allclose(transformed['depth'],s*depth)
    assert quality['normalized_center_rms']<1e-6


def test_vggt_degenerate_overlap_rejected():
    ext=np.tile(np.c_[np.eye(3),np.zeros(3)],(6,1,1))
    with pytest.raises(ValueError,match='Degenerate'):camera_alignment(ext,ext)


def test_foreground_preserves_alpha_and_opaque_fallback():
    rgba=np.full((40,40,4),255,'u1');rgba[:10,:,3]=0
    mask,source=object_mask(Image.fromarray(rgba));assert source=='supplied_alpha' and not mask[:10].any()
    mask,source=object_mask(Image.new('RGB',(40,40),'white'));assert mask.min()==255


def test_vggt_padding_preserves_projection_and_excludes_mask_border(tmp_path):
    from web.vggt_batches import pad_predictions,preprocess_masks
    data=dict(images=np.zeros((2,392,518,3),'f4'),world_points=np.zeros((2,392,518,3),'f4'),world_points_conf=np.ones((2,392,518),'f4'),depth=np.ones((2,392,518,1),'f4'),depth_conf=np.ones((2,392,518),'f4'),intrinsic=np.tile(np.eye(3),(2,1,1)))
    padded=pad_predictions(data,518)
    assert padded['images'].shape==(2,518,518,3)
    assert np.all(padded['intrinsic'][:,1,2]==63)
    assert not padded['depth_conf'][:,:63].any()
    source=tmp_path/'mask.png';Image.new('L',(800,600),255).save(source)
    mask=preprocess_masks([source],518)
    assert not mask[:,:63].any() and mask[:,63:455].all() and not mask[:,455:].any()


def test_feature_retrieval_recognizes_distant_texture_revisits():
    rng=np.random.default_rng(9)
    images=[np.full((16,16,3),40,'u1') for _ in range(60)]
    prototypes=rng.uniform(0,1,(12,128)).astype('f4')
    descriptors=[np.repeat(prototypes[i%12][None],80,axis=0)+rng.normal(0,.01,(80,128)).astype('f4') for i in range(60)]
    pairs=select_pairs(images,features=descriptors)
    assert (0,24) in pairs and (5,41) in pairs
    assert len(pairs)<900


def test_dataset_selection_keeps_later_views_and_natural_order(tmp_path):
    from web.seed_datasets import select_photos
    rng=np.random.default_rng(1)
    for i in range(90):Image.fromarray(rng.integers(0,255,(20,20,3),dtype='u1')).save(tmp_path/f'r_{i}.png')
    photos=select_photos(tmp_path)
    ids=[int(p.stem[2:]) for p in photos]
    assert len(photos)==60 and ids==sorted(ids) and ids[0]<2 and ids[-1]>=78


def test_foreground_crop_preserves_original_pixel_calibration(tmp_path):
    import json,yaml
    from web.foreground import prepare_foreground
    (tmp_path/'uploads').mkdir();(tmp_path/'input/images').mkdir(parents=True)
    rgba=np.zeros((200,400,4),'u1');rgba[40:160,100:300,:3]=[20,100,220];rgba[40:160,100:300,3]=255
    Image.fromarray(rgba).save(tmp_path/'uploads/000_car.png')
    Image.fromarray(rgba[:,:,:3]).resize((200,100)).save(tmp_path/'input/images/000_car.jpg')
    K=np.array([[300.,0,100],[0,300,50],[0,0,1]])
    (tmp_path/'input/intrinsics.yaml').write_text(yaml.safe_dump(dict(images={'000_car.jpg':dict(K=K.tolist(),image_size=[200,100],source='supplied')})))
    (tmp_path/'job.json').write_text(json.dumps(dict(max_side=200)))
    original=(tmp_path/'uploads/000_car.png').read_bytes()
    prepare_foreground(tmp_path)
    row=json.loads((tmp_path/'foreground.json').read_text())[0];record=yaml.safe_load((tmp_path/'input/intrinsics.yaml').read_text())['images']['000_car.jpg']
    box=row['crop_original'];w,h=row['prepared_size'];expected=K.copy();expected[:2]*=2;expected[:2,2]-=box[:2];expected[0]*=w/(box[2]-box[0]);expected[1]*=h/(box[3]-box[1])
    assert np.allclose(record['K'],expected)
    assert record['image_size']==[w,h] and w<=200 and h<=200
    assert (tmp_path/'uploads/000_car.png').read_bytes()==original
    assert Image.open(tmp_path/'input_masks/000_car.jpg.png').size==(w,h)


def test_robust_triangulation_drops_one_outlier_without_discarding_point():
    from sfm.incremental import triangulate_track,filter_points
    from sfm.reconstruction import Reconstruction,Camera
    K=np.array([[500.,0,200],[0,500,150],[0,0,1.]])
    X=np.array([.1,.2,4.]);recon=Reconstruction();keypoints=[]
    for i,center in enumerate(([-1,0,0],[0,0,0],[1,0,0],[0,1,0])):
        t=-np.array(center,float);recon.cameras[i]=Camera(i,str(i),K.copy(),np.eye(3),t,True)
        q=K@(X+t);xy=q[:2]/q[2]
        if i==3:xy+=np.array([80.,-65.])
        keypoints.append(xy[None])
    track={i:0 for i in range(4)};config=dict(max_reproj_px=2.,min_angle_deg=1.)
    assert triangulate_track(recon,dict(track),keypoints,config) is None
    recovered=triangulate_track(recon,track,keypoints,dict(config,robust=True))
    assert np.allclose(recovered,X) and set(track)=={0,1,2}
    recon.points[0]=X;recon.tracks[0]={i:0 for i in range(4)}
    assert filter_points(recon,keypoints,2.,True)==0 and set(recon.tracks[0])=={0,1,2}


def test_colmap_mixed_sizes_keep_calibration_and_valid_native_frames(tmp_path):
    import importlib.util,json,subprocess,sys,yaml
    from pathlib import Path
    if importlib.util.find_spec('pycolmap') is None:pytest.skip('Optional COLMAP runtime')
    images=tmp_path/'images';images.mkdir();rng=np.random.default_rng(14);records={}
    for i,(w,h) in enumerate([(160,120),(160,120),(120,160),(120,160)]):
        name=f'{i}.jpg';Image.fromarray(rng.integers(0,255,(h,w,3),dtype='u1')).save(images/name)
        records[name]=dict(K=[[180,0,w/2],[0,180,h/2],[0,0,1]],image_size=[w,h],source='user_camera_matrix_at_original_resolution')
    (tmp_path/'intrinsics.yaml').write_text(yaml.safe_dump(dict(images=records)))
    root=Path(__file__).resolve().parents[1];out=tmp_path/'colmap'
    # Native feature/mapping calls stay isolated from Torch/OpenMP test workers.
    process=subprocess.run([sys.executable,str(root/'code/web/colmap_job.py'),'--images',str(images),'--out',str(out),'--object-quality'],capture_output=True,text=True,timeout=30)
    assert 'frame.HasDataId' not in process.stderr
    reader='import pycolmap,json,sys; d=pycolmap.Database.open(sys.argv[1]); print(json.dumps({i.name:d.read_camera(i.camera_id).calibration_matrix().tolist() for i in d.read_all_images()}))'
    actual=json.loads(subprocess.check_output([sys.executable,'-c',reader,str(out/'database.db')],text=True,timeout=10))
    assert set(actual)==set(records)
    for name,K in actual.items():assert np.allclose(K,records[name]['K'])
