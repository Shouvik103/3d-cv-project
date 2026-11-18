"""Analytic geometry gates, written before running any real mesh scene."""
from pathlib import Path
import json, subprocess, sys
import numpy as np
import pytest


def sphere_frames(size=96, n=12):
    from mesh.synthetic import analytic_scene
    return analytic_scene('sphere', size=size, views=n)


def test_robust_inverse_depth_alignment():
    from mesh.align_depth import fit_inverse_depth_affine, aligned_depth
    rng=np.random.default_rng(4);z=rng.uniform(1,5,600);s,b=1.8,.13;d=(1/z-b)/s
    d[:30]=rng.uniform(-2,2,30)
    fit=fit_inverse_depth_affine(d,z,seed=2)
    assert abs(fit[0]/s-1)<.02 and abs(fit[1]/b-1)<.02
    assert np.median(abs(aligned_depth(d[30:],*fit[:2])-z[30:])/z[30:])<.02
    with pytest.raises(ValueError):fit_inverse_depth_affine(np.ones(50),np.ones(50))


@pytest.mark.parametrize('shape',['sphere','box'])
def test_exact_depth_tsdf_and_pose_convention(shape):
    from mesh.synthetic import analytic_scene
    from mesh.tsdf_fusion import fuse
    from mesh.mesh_eval import mesh_statistics
    frames=analytic_scene(shape,size=112,views=12);voxel=.045
    mesh,_=fuse(frames,voxel,4*voxel,8)
    # The production mesh includes cleanup of tiny TSDF silhouette fragments.
    from mesh.mesh_clean import clean
    mesh,_=clean(mesh,(np.full(3,-1.25),np.full(3,1.25)))
    v=np.asarray(mesh.vertices)
    error=abs(np.linalg.norm(v,axis=1)-1) if shape=='sphere' else abs(np.abs(v).max(1)-.75)
    assert np.median(error)<.5*voxel and np.quantile(error,.99)<1.5*voxel
    assert mesh_statistics(mesh)['components']==1
    wrong=[dict(f,R=f['R'].T,t=-f['R'].T@f['t']) for f in frames]
    bad,_=fuse(wrong,voxel,4*voxel,8)
    bv=np.asarray(bad.vertices)
    assert not len(bv) or np.median(abs(np.linalg.norm(bv,axis=1)-1))>3*voxel


def test_raycast_depth_normals_and_colors():
    import open3d as o3d
    from mesh.rendering import render
    f=sphere_frames(size=80,n=1)[0]
    m=o3d.geometry.TriangleMesh.create_sphere(1,resolution=160);m.compute_vertex_normals();m.paint_uniform_color([.2,.4,.8])
    result=render(m,f['K'],f['R'],f['t'],80,80)
    good=(f['depth']>0)&result['mask']
    # Interior rays avoid silhouettes, where any finite tessellation has larger errors.
    good&=np.linalg.norm(result['normals'],axis=2)>.9
    good&=abs(result['normals'][:,:,2])<2 # finite normal guard
    truth=f['depth'];interior=good & (truth<2.55)
    assert np.median(abs(result['depth'][interior]-truth[interior])/truth[interior])<1e-3
    yy,xx=np.indices(truth.shape);rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(f['K']).T
    world=(rays*truth[:,:,None]-f['t'])@f['R'];normal=world/np.maximum(np.linalg.norm(world,axis=2,keepdims=True),1e-9)
    angles=np.degrees(np.arccos(np.clip((normal*result['normals']).sum(2),-1,1)))
    assert np.median(angles[interior])<2
    assert np.max(abs(result['color'][interior].mean(0)-[51,102,204]))<1


def test_clean_export_and_headless_viewer(tmp_path):
    import open3d as o3d
    from mesh.mesh_clean import clean
    from mesh.mesh_export import export_meshes,read_mesh
    from mesh.mesh_eval import compare_meshes,mesh_statistics
    m=o3d.geometry.TriangleMesh.create_sphere(1,70);m.compute_vertex_normals();m.paint_uniform_color([.2,.4,.8])
    tiny=o3d.geometry.TriangleMesh.create_sphere(.02,10).translate([1.5,0,0]);tiny.paint_uniform_color([.2,.4,.8]);m+=tiny
    full,light=clean(m,(np.array([-2]*3),np.array([2]*3)),light_target=2500)
    assert mesh_statistics(full)['components']==1
    assert compare_meshes(full,light,samples=15000,seed=2)['chamfer_mean']<.005*3.464
    export_meshes(full,light,tmp_path)
    for ext in ['ply','obj','glb']:
        loaded=read_mesh(tmp_path/('mesh_light.'+ext))
        assert len(loaded.vertices)==len(light.vertices)
        assert np.allclose(np.asarray(loaded.vertex_colors).mean(0),[.2,.4,.8],atol=1/255)
    script=Path(__file__).resolve().parents[1]/'code/viz/view_mesh.py'
    subprocess.run([sys.executable,str(script),'--mesh',str(tmp_path/'mesh_light.glb'),'--screenshot',str(tmp_path/'view.png')],check=True,cwd=tmp_path)
    assert (tmp_path/'view.png').stat().st_size>1000


def test_adapter_convention_and_resolution(tmp_path):
    from mesh.synthetic import write_sample
    from mesh.adapters import load_scene,check_convention
    write_sample(tmp_path,size=80,views=6)
    scene=load_scene(tmp_path);report=check_convention(scene,tmp_path/'check')
    assert report['median_reprojection_px']<1e-5
    assert report['passed']
    poses=json.loads((tmp_path/'poses.json').read_text());poses['image_size'][poses['image_names'][0]]=[2,2];(tmp_path/'poses.json').write_text(json.dumps(poses))
    with pytest.raises(ValueError,match='size'):load_scene(tmp_path)


def test_no_track_convention_gate_writes_serializable_status(tmp_path):
    from mesh.synthetic import write_sample
    from mesh.adapters import load_scene,check_convention
    write_sample(tmp_path/'sample',size=64,views=6)
    scene=load_scene(tmp_path/'sample');scene.tracks=None
    report=check_convention(scene,tmp_path/'check')
    assert type(report['passed']) is bool and report['passed']
    assert json.loads((tmp_path/'check/convention.json').read_text())['passed'] is True


def test_cached_cli_heldout_and_relocation(tmp_path):
    from mesh.synthetic import write_sample
    from mesh.mesh_export import read_mesh
    write_sample(tmp_path/'sample',size=80,views=12)
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,str(root/'code/mesh/run_mesh.py'),'--scene',str(tmp_path/'sample'),'--out',str(tmp_path/'mesh'),'--config',str(root/'configs/mesh_sample.yaml')]
    subprocess.run(cmd,check=True,cwd=tmp_path)
    result=json.loads((tmp_path/'mesh/summary.json').read_text())
    assert result['frames_fused']>=6 and result['heldout']['views']==2
    mesh=read_mesh(tmp_path/'mesh/mesh_light.ply')
    assert len(mesh.triangles)>100
    assert np.median(abs(np.linalg.norm(np.asarray(mesh.vertices),axis=1)-1))<.04
    assert not set(result['fusion_images'])&set(result['heldout_images'])
    assert result['timing_seconds']['total']<180


def test_sgbm_returns_reference_camera_metric_depth():
    import cv2
    from mesh.depth_sgbm import stereo_pair
    rng=np.random.default_rng(0)
    left=cv2.GaussianBlur(rng.integers(0,255,(120,160,3),dtype='u1'),(3,3),0)
    right=np.zeros_like(left);right[:,:-11]=left[:,11:]
    class Plane:
        names=['left.png','right.png']
        K=[np.array([[220.,0,80],[0,220,60],[0,0,1]])]*2
        R=[np.eye(3)]*2
        t=[np.zeros(3),np.array([-.15,0,0])]
        points=np.array([[x,y,3.] for x in np.linspace(-.7,.7,15) for y in np.linspace(-.5,.5,12)])
        def image(self,i):return [left,right][i]
        def mask(self,i):return np.ones((120,160),bool)
    frame,_=stereo_pair(Plane(),0,1,long_side=160,num_disparities=32)
    z=frame['depth'];assert (z>0).sum()>1000
    assert np.median(abs(z[z>0]-3)/3)<.03


def test_consistency_on_more_than_32767_pixels():
    from mesh.filters import consistency_filter
    f=dict(depth=np.ones((160,320),'f4'),K=np.array([[300.,0,160],[0,300,80],[0,0,1]]),R=np.eye(3),t=np.zeros(3))
    frames=consistency_filter([f,f,f])
    assert all(np.count_nonzero(g['depth'])==160*320 for g in frames)
