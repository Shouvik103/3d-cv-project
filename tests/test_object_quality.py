"""Object quality invariants: track preservation, calibration, matching and depth."""
from pathlib import Path
import numpy as np
import pytest
from sfm.tracks import build_tracks
from sfm.pair_selection import select_pairs
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








def test_foreground_preserves_alpha_and_opaque_fallback():
    rgba=np.full((40,40,4),255,'u1');rgba[:10,:,3]=0
    mask,source=object_mask(Image.fromarray(rgba));assert source=='supplied_alpha' and not mask[:10].any()
    mask,source=object_mask(Image.new('RGB',(40,40),'white'));assert mask.min()==255




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
    assert len(photos)==60 and ids==sorted(ids) and ids[0]<2 and ids[-1]>=68


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


def test_depth_samples_reproject_to_observed_pixels_and_never_fill_to_cap():
    from mesh.depth_cloud import depth_samples
    K=np.array([[120.,0,2.],[0,120.,2.],[0,0,1.]])
    R=np.array([[0.,-1.,0],[1.,0,0],[0,0,1.]])
    color=np.arange(5*6*3,dtype='u1').reshape(5,6,3);depth=np.zeros((5,6),'f4');depth[1:4,1:5]=3.
    frame=dict(index=9,depth=depth,color=color,K=K,R=R,t=np.array([.4,-.8,.1]))
    points,colors,meta=depth_samples([frame],limit=1000)
    assert len(points)==12 and meta['exported_points']==12 and meta['available_pixels']==12
    camera=points@R.T+frame['t'];projected=camera@K.T;uv=projected[:,:2]/projected[:,2:]
    assert np.allclose(uv,np.rint(uv),atol=2e-6)
    assert np.allclose(camera[:,2],3.)
    assert np.array_equal(colors,color[np.rint(uv[:,1]).astype(int),np.rint(uv[:,0]).astype(int)])
    small,_,meta=depth_samples([frame,dict(frame,index=10)],limit=10)
    assert len(small)==10 and all(r['exported_points']==5 for r in meta['views'])


def test_mesh_only_worker_reuses_cameras_and_exports_real_depth_samples(tmp_path,monkeypatch):
    import json,shutil,yaml
    from mesh.synthetic import write_sample,analytic_scene
    from prepare_data import prepare
    from web.worker import run_job
    import web.worker as worker
    scene=tmp_path/'sample';write_sample(scene,size=80,views=6)
    job=tmp_path/'job';shutil.copytree(scene/'images',job/'uploads');prepare(job/'uploads',job/'input/images',80)
    recon=job/'reconstruction';recon.mkdir();poses=json.loads((scene/'poses.json').read_text());names={n:Path(n).stem+'.jpg' for n in poses['image_names']}
    poses['image_names']=[names[n] for n in poses['image_names']];poses['registered']=[names[n] for n in poses['registered']]
    for key in ('K','R','t','image_size'):poses[key]={names[n]:v for n,v in poses[key].items()}
    (recon/'poses.json').write_text(json.dumps(poses));original=(recon/'poses.json').read_bytes()
    observations=json.loads((scene/'observations.json').read_text());observations={k:{names[n]:uv for n,uv in track.items()} for k,track in observations.items()}
    (recon/'observations.json').write_text(json.dumps(observations));shutil.copyfile(scene/'points.ply',recon/'points.ply')
    (recon/'summary.json').write_text(json.dumps(dict(registered=6,total_images=6,points=len(observations),runtime_seconds=23.,median_reprojection_px=0.)))
    metadata=dict(method='ours',quality_profile='object',reuse_geometry=True,make_mesh=True,mesh_quality='balanced',depth_backend='mono',max_side=80,uploads=[])
    (job/'job.json').write_text(json.dumps(metadata))
    def no_inference(*args,**kwargs):raise AssertionError('A mesh-only rebuild must not invoke SfM')
    monkeypatch.setattr(worker,'run_sfm',no_inference)
    from mesh.depth_mono import MonoDepth
    fixture=analytic_scene(size=80,views=6,pixel_center=0)
    def cached_fixture(self,loaded,i,cache_only=False):
        depth=fixture[i]['depth'];relative=np.zeros_like(depth);good=depth>0;relative[good]=(1/depth[good]-.1)/1.7
        return relative,dict(cache_hit=True,provenance='Analytic fixture, not a learned prediction')
    monkeypatch.setattr(MonoDepth,'predict',cached_fixture)
    assert run_job(job)==0
    state=json.loads((job/'state.json').read_text());assert state['status']=='completed'
    assert state['metrics']['geometry_reused'] is True and state['metrics']['reconstruction_seconds']==0
    assert state['metrics']['original_reconstruction_seconds']==23.
    assert state['artifacts']['depth_sample_cloud']=='mesh/depth_samples.ply'
    assert (recon/'poses.json').read_bytes()==original
    preview=json.loads((job/'cloud.json').read_text());summary=json.loads((job/'mesh/summary.json').read_text())
    assert preview['total_points']==summary['depth_samples']['exported_points']>100
    assert summary['depth_samples']['available_pixels']>=preview['total_points']






def test_quality_inputs_repeat_rotation_and_calibration_without_using_poses(tmp_path):
    import json,yaml
    from web.quality_input import prepare_quality_input
    job=tmp_path/'job';(job/'uploads').mkdir(parents=True)
    rgba=np.full((120,80,4),255,'u1');rgba[:20,:,3]=0;rgba[20:100,20:60,:3]=[20,80,180]
    Image.fromarray(rgba).save(job/'uploads/a.png');original=(job/'uploads/a.png').read_bytes()
    K=np.array([[100.,0,40],[0,110.,60],[0,0,1.]])
    (job/'job.json').write_text(json.dumps(dict(method='ours',camera_matrix=K.tolist())))
    prepare_quality_input(job,80)
    record=yaml.safe_load((job/'input/intrinsics.yaml').read_text())['images']['a.jpg'];w,h=record['image_size']
    assert record['rotation_degrees']==90 and [w,h]==[80,53]
    point=np.array([.1,.2,3.]);pixel=K@point;pixel=pixel[:2]/pixel[2]
    expected=np.array([119-pixel[1],pixel[0]])*np.array([w/120,h/80])
    Q=np.array([[0.,-1,0],[1.,0,0],[0,0,1.]])
    actual=np.array(record['K'])@(Q@point);assert np.allclose(actual[:2]/actual[2],expected)
    prepare_quality_input(job,60,target=job/'mesh_input',record=False)
    matching=json.loads((job/'foreground.json').read_text())[0];assert matching['prepared_size']==[80,53]
    assert (job/'uploads/a.png').read_bytes()==original
    assert (job/'input/feature_masks/a.jpg.png').exists()


def test_ranked_eight_point_handles_weak_overlap_and_outliers():
    from sfm.robust_matching import verify_ranked,hypotheses
    from sfm.geometry import sampson_distance
    rng=np.random.default_rng(93);X=rng.normal(0,.6,(120,3));X[:,2]+=5
    K=np.array([[900.,0,400],[0,900.,300],[0,0,1.]])
    def project(x):
        q=x@K.T;return q[:,:2]/q[:,2:]
    a=project(X);b=project(X+[-.5,.02,.04]);a+=rng.normal(0,.15,a.shape);b+=rng.normal(0,.15,b.shape)
    # Strong descriptor matches occur first, but some of them remain wrong.
    b[8:14]=rng.uniform([0,0],[800,600],(6,2));b[40:]=rng.uniform([0,0],[800,600],(80,2))
    F,mask=verify_ranked(a,b,threshold=1,iterations=2000,seed=4)
    true=np.r_[np.arange(8),np.arange(14,40)]
    assert mask[true].sum()>=30 and mask[40:].sum()<8
    assert np.median(sampson_distance(F,a[true],b[true]))<.2
    Fs,valid=hypotheses(a[true[:8]][None],b[true[:8]][None]);assert valid[0]
    assert abs(np.linalg.det(Fs[0]))<1e-10
    again,m2=verify_ranked(a,b,threshold=1,iterations=2000,seed=4)
    assert np.allclose(again,F) and np.array_equal(m2,mask)


def test_partial_diagnostics_separate_track_support_from_sharpness(tmp_path):
    import json
    from web.diagnostics import inspect_registration
    (tmp_path/'reconstruction').mkdir();(tmp_path/'input/images').mkdir(parents=True)
    names=['a.jpg','b.jpg','c.jpg','d.jpg']
    for n in names:Image.new('RGB',(20,20),'white').save(tmp_path/'input/images'/n)
    def save(n,value):(tmp_path/'reconstruction'/n).write_text(json.dumps(value))
    save('poses.json',dict(image_names=names,registered=names[:2]))
    save('matching_report.json',dict(checked_pairs=6,verified_pairs=[dict(images=names[:2],inliers=80),dict(images=names[2:],inliers=60)]))
    save('registration_checks.json',{'c.jpg':dict(shared_3d_tracks=0,reason='Insufficient shared 3D tracks'),'d.jpg':dict(shared_3d_tracks=20,reason='PnP pose check failed')})
    out=inspect_registration(tmp_path,{})
    assert list(map(len,out['components']))==[2,2] and out['missing']==names[2:]
    assert '1 missing views lacked' in out['explanation'] and '1 failed PnP' in out['explanation']
    assert out['views'][2]['shared_3d_tracks']==0 and out['views'][3]['verified_links_to_registered']==0


def test_common_adapter_accepts_explicit_distortion_for_multiple_radial_cameras(tmp_path):
    import json
    from mesh.adapters import load_scene
    from sfm.io_utils import save_ply
    images=tmp_path/'images';images.mkdir();names=['a.jpg','b.jpg']
    for n in names:Image.new('RGB',(40,30),'white').save(images/n)
    K=[[60.,0,20],[0,60.,15],[0,0,1]]
    poses=dict(image_names=names,registered=names,K={n:K for n in names},R={n:np.eye(3).tolist() for n in names},t={n:[0,0,0] for n in names},distortion={n:[-.02,0,0,0,0] for n in names})
    (tmp_path/'poses.json').write_text(json.dumps(poses));(tmp_path/'summary.json').write_text(json.dumps(dict(cameras=[dict(model='SIMPLE_RADIAL'),dict(model='SIMPLE_RADIAL')])))
    save_ply(np.array([[0,0,2],[.1,0,2],[0,.1,2],[.1,.1,2.2]]),None,tmp_path/'points.ply')
    scene=load_scene(tmp_path);assert len(scene.K)==2 and all(np.isclose(d[0],-.02) for d in scene.distortion)
