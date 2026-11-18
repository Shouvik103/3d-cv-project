"""Upload boundaries and persisted-job behavior for the localhost workbench."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import io
import json
import threading
import subprocess
import sys

import pytest
from PIL import Image
from web.server import Workbench, make_handler
from web.worker import write_state


@pytest.fixture
def workbench_api(tmp_path):
    app = Workbench(tmp_path / 'jobs')
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, payload=None, headers=None):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        default = {'X-Reconstruction-UI': '1'}
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode()
            default['Content-Type'] = 'application/json'
        connection.request(method, path, payload, default | (headers or {}))
        response = connection.getresponse()
        data = response.read()
        code, content_type = response.status, response.getheader('Content-Type')
        connection.close()
        return code, json.loads(data) if content_type == 'application/json' else data

    yield app, request
    server.shutdown()
    server.server_close()
    thread.join()
    app.shutdown()


def test_upload_validation_and_request_boundaries(workbench_api):
    app, request = workbench_api
    assert request('POST', '/api/jobs', {}, {'Origin': 'https://unrelated.example'})[0] == 403
    assert request('GET', '/', headers={'Host': 'unrelated.example'})[0] == 403
    code, job = request('POST', '/api/jobs', {'name': '<unsafe title>', 'make_mesh': False})
    assert code == 201
    job_id = job['id']
    assert request('POST', f'/api/jobs/{job_id}/start', {})[0] == 400
    assert request('PUT', f'/api/jobs/{job_id}/images/0?name=broken.jpg', b'not an image')[0] == 400
    buffer = io.BytesIO()
    Image.new('RGB', (20, 20), (20, 80, 40)).save(buffer, format='PNG')
    assert request('PUT', f'/api/jobs/{job_id}/images/0?name=../photo.png', buffer.getvalue())[0] == 200
    code, saved = request('GET', f'/api/jobs/{job_id}')
    assert code == 200 and saved['uploads'][0]['stored'] == '000_photo.png'
    assert request('GET', f'/api/jobs/{job_id}/images/0')[0] == 200
    assert request('PUT', f'/api/jobs/{job_id}/images/0?name=other.png', buffer.getvalue())[0] == 400
    assert request('GET', '/static/../../README.md')[0] == 400
    assert request('GET', f'/api/jobs/{job_id}/files/secret')[0] == 400
    write_state(app.folder(job_id), artifacts={'secret': '../../outside.txt'})
    assert request('GET', f'/api/jobs/{job_id}/files/secret')[0] == 400
    assert request('POST', f'/api/jobs/{job_id}/cancel', {})[1]['status'] == 'cancelled'


def test_restart_preserves_finished_runs_and_marks_interrupted(tmp_path):
    app = Workbench(tmp_path)
    active = app.create({})
    finished = app.create({})
    write_state(app.folder(active['id']), status='running')
    write_state(app.folder(finished['id']), status='completed', artifacts={'cloud_view': 'cloud.json'})
    app.shutdown()
    restarted = Workbench(tmp_path)
    try:
        assert restarted.describe(restarted.folder(active['id']))['status'] == 'interrupted'
        assert restarted.describe(restarted.folder(finished['id']))['status'] == 'completed'
    finally:
        restarted.shutdown()


def test_geometry_keeps_mesh_connectivity_and_caps_point_preview(tmp_path):
    import numpy as np
    import open3d as o3d
    from web.geometry import export_geometry
    mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=10)
    mesh.paint_uniform_color([.2, .4, .8])
    source = tmp_path / 'mesh.ply'
    o3d.io.write_triangle_mesh(str(source), mesh)
    before = source.read_bytes()
    data = export_geometry(source, tmp_path / 'mesh.json', kind='mesh')
    assert data['triangles'] == len(mesh.triangles)
    assert np.array_equal(np.array(data['indices']).reshape(-1, 3), np.asarray(mesh.triangles))
    assert source.read_bytes() == before
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    cloud.colors = mesh.vertex_colors
    path = tmp_path / 'cloud.ply'
    o3d.io.write_point_cloud(str(path), cloud)
    points = export_geometry(path, tmp_path / 'cloud.json', limit=100)
    assert points['total_points'] == len(mesh.vertices)
    assert points['displayed_points'] == 100
    assert points['indices'] is None


def test_delete_stops_running_job_and_moves_files_to_trash(workbench_api):
    app, request = workbench_api
    job = app.create({'name': 'Delete test'})
    folder = app.folder(job['id'])
    (folder / 'saved-result.txt').write_text('keep recoverable')
    process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    app.processes[job['id']] = process
    write_state(folder, status='running')
    monitor = app.pool.submit(app.monitor, folder, process)
    code, result = request('DELETE', '/api/jobs/' + job['id'])
    assert code == 200 and result['deleted']
    monitor.result(timeout=10)
    assert process.poll() is not None
    assert request('GET', '/api/jobs/' + job['id'])[0] == 400
    assert not request('GET', '/api/jobs')[1]['jobs']
    archived = list((app.root / '.trash').glob(job['id'] + '-*'))
    assert (archived[0] / 'saved-result.txt').read_text() == 'keep recoverable'


def test_delete_queued_job_never_starts(workbench_api):
    app, request = workbench_api
    gate = threading.Event()
    blocked = app.pool.submit(gate.wait, 10)
    try:
        job = app.create({})
        folder = app.folder(job['id'])
        metadata = json.loads((folder / 'job.json').read_text())
        metadata['uploads'] = [dict(index=i, stored=f'{i}.png', name=f'{i}.png', bytes=1) for i in range(3)]
        (folder / 'job.json').write_text(json.dumps(metadata))
        app.start(job['id'])
        assert request('DELETE', '/api/jobs/' + job['id'])[0] == 200
    finally:
        gate.set()
        blocked.result(timeout=10)
    app.pool.submit(lambda: None).result(timeout=10)
    archived = next((app.root / '.trash').glob(job['id'] + '-*'))
    assert not (archived / 'run.log').exists()


def test_delete_existing_example_persists_without_removing_sources(workbench_api, monkeypatch):
    import web.server as server
    app, request = workbench_api
    monkeypatch.setattr(server, 'ROOT', app.root.parent)
    source = server.ROOT / 'results/mesh/car_sgbm/mesh_light.ply'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'original geometry')
    assert any(item['id'] == 'car' for item in request('GET', '/api/jobs')[1]['examples'])
    assert request('DELETE', '/api/examples/car')[0] == 200
    assert not any(item['id'] == 'car' for item in request('GET', '/api/jobs')[1]['examples'])
    assert source.read_bytes() == b'original geometry'
    restarted = Workbench(app.root)
    try:
        assert not any(item['id'] == 'car' for item in restarted.examples())
    finally:
        restarted.shutdown()


def test_legacy_progress_counts_attempted_pairs_not_only_successes(tmp_path):
    from web.progress import legacy_progress
    metadata = dict(uploads=[dict(stored=f'{i:03d}_photo.png') for i in range(200)])
    (tmp_path / 'run.log').write_text('Pair 011_photo.jpg/022_photo.jpg: 21 inliers, H/F 0.38\n')
    state = dict(status='running', stage='reconstructing', progress=15)
    report = legacy_progress(tmp_path, metadata, state)
    assert report['progress'] > 22
    assert '2,145/19,900' in report['message']
    measured = dict(state, progress=32, progress_units=dict(done=10000,total=19900))
    assert legacy_progress(tmp_path, metadata, measured) == measured


def test_sgbm_mesh_reports_depth_and_fusion_progress(tmp_path):
    import yaml
    from mesh.run_mesh import run, merge
    root = Path(__file__).resolve().parents[1]
    config = merge(yaml.safe_load((root / 'configs/mesh_default.yaml').read_text()),
                   yaml.safe_load((root / 'configs/mesh_sgbm.yaml').read_text()))
    updates = []
    summary = run(root / 'sample_data/scene_small', tmp_path / 'mesh', config,
                  backend='sgbm', progress=lambda phase,done,total,message: updates.append((phase,done,total)))
    assert summary['light']['faces'] > 100
    count = len(summary['fusion_images'])
    depth = [(done,total) for phase,done,total in updates if phase == 'depth']
    assert depth[0][0] == 0 and depth[-1][0] == depth[-1][1]
    assert depth[-1][0] >= count
    assert {'depth','filtering','fusion','cleanup','mesh_preview'} <= {phase for phase,_,_ in updates}


@pytest.mark.parametrize('quality,depth_side,voxel_div',[('balanced',640,192),('high',960,256),('ultra',1280,384)])
def test_mesh_quality_is_validated_and_saved(workbench_api, quality, depth_side, voxel_div):
    from web.settings import mesh_quality
    app,request = workbench_api
    status,job = request('POST','/api/jobs',dict(mesh_quality=quality,max_side=800))
    assert status == 201 and job['mesh_quality'] == quality
    assert mesh_quality(job)['depth_side'] == depth_side
    assert mesh_quality(job)['voxel_div'] == voxel_div
    assert request('POST','/api/jobs',dict(mesh_quality='unbounded'))[0] == 400


def test_mesh_images_recover_original_resolution_without_changing_geometry(tmp_path):
    import shutil
    import numpy as np
    from mesh.synthetic import write_sample
    from mesh.adapters import load_scene, check_convention
    from web.mesh_input import prepare_mesh_scene
    source = tmp_path / 'source'
    write_sample(source, size=80, views=6)
    job = tmp_path / 'job'
    shutil.copytree(source / 'images', job / 'uploads')
    reconstruction = job / 'reconstruction'
    reconstruction.mkdir()
    (job / 'input/images').mkdir(parents=True)
    poses = json.loads((source / 'poses.json').read_text())
    observations = json.loads((source / 'observations.json').read_text())
    # UI preparation normalizes upload names to JPEG before SfM.
    names={name:Path(name).stem+'.jpg' for name in poses['image_names']}
    poses['image_names']=[names[name] for name in poses['image_names']]
    poses['registered']=[names[name] for name in poses['registered']]
    for key in ('K','R','t','image_size'):
        poses[key]={names[name]:value for name,value in poses[key].items()}
    for original,name in names.items():
        with Image.open(source / 'images' / original) as image:
            image.resize((40,40)).save(job / 'input/images' / name)
        K = np.array(poses['K'][name]);K[:2] *= .5
        poses['K'][name] = K.tolist();poses['image_size'][name]=[40,40]
    for track in observations.values():
        renamed={names[name]:(np.array(uv)*.5).tolist() for name,uv in track.items()}
        track.clear();track.update(renamed)
    (reconstruction / 'poses.json').write_text(json.dumps(poses))
    (reconstruction / 'observations.json').write_text(json.dumps(observations))
    shutil.copyfile(source / 'points.ply', reconstruction / 'points.ply')
    original=(reconstruction / 'poses.json').read_bytes()
    scene_path = prepare_mesh_scene(job,960)
    scene = load_scene(scene_path)
    assert scene.image(0).shape[:2] == (80,80) # Uses original pixels, with no upscaling.
    truth=json.loads((source / 'poses.json').read_text())
    assert np.allclose(scene.K[0],truth['K'][Path(scene.names[0]).stem+'.png'])
    assert check_convention(scene)['median_reprojection_px'] < 1e-5
    assert (reconstruction / 'poses.json').read_bytes() == original
    assert (scene_path / 'reconstruction/points.ply').read_bytes() == (source / 'points.ply').read_bytes()


@pytest.mark.parametrize('long_side',[960,1280])
def test_high_resolution_stereo_keeps_depth_scale(long_side):
    import cv2
    import numpy as np
    from mesh.depth_sgbm import stereo_pair
    rng=np.random.default_rng(4)
    left=cv2.GaussianBlur(rng.integers(0,255,(960,1280,3),dtype='u1'),(3,3),0)
    right=np.zeros_like(left);right[:,:-55]=left[:,55:]
    class Plane:
        names=['left.jpg','right.jpg']
        K=[np.array([[1100.,0,640],[0,1100,480],[0,0,1]])]*2
        R=[np.eye(3)]*2
        t=[np.zeros(3),np.array([-.15,0,0])]
        points=np.array([[x,y,3.] for x in np.linspace(-.8,.8,15) for y in np.linspace(-.5,.5,12)])
        def image(self,i):return [left,right][i]
        def mask(self,i):return np.ones((960,1280),bool)
    frame,_=stereo_pair(Plane(),0,1,long_side=long_side)
    depth=frame['depth']
    assert depth.shape == (long_side*3//4,long_side)
    assert (depth>0).sum()>10000
    assert np.median(abs(depth[depth>0]-3)/3)<.03
