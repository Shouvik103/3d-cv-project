"""Local-only image upload API and reconstruction job queue (standard library)."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import hashlib
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit, parse_qs
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web.worker import write_state
from web.progress import legacy_progress
from web.settings import mesh_quality,method_settings,METHOD_NAMES,MAX_UPLOAD_VIEWS
from web.research import catalog, asset_path, run_analysis

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / 'static'
MAX_IMAGE = 25 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
TERMINAL = {'completed', 'partial', 'failed', 'cancelled', 'interrupted'}
EXAMPLES = {
    'car': dict(name='Car · 60 views', folder='car_sgbm', poses='outputs/cars_curated_60/colmap_car_2400/poses.json',
                cloud='outputs/cars_curated_60/car_foreground/car.ply'),
    'chair': dict(name='Chair · 60 views', folder='chair_sgbm_wide_roi', poses='outputs/chair_test/colmap/poses.json',
                  cloud='outputs/chair_test/foreground/chair.ply'),
}


def example_source(example, kind):
    """Use the original object cloud, rather than the smaller TSDF mesh cloud."""
    if kind == 'cloud':
        return ROOT / example['cloud']
    return ROOT / 'results/mesh' / example['folder'] / 'mesh_light.ply'


def example_cache(root, key, example, kind):
    source = example_source(example, kind)
    poses = ROOT / example['poses']
    # Include source identity, not just timestamps: earlier cached clouds came
    # from mesh/dense.ply and must never replace these original reconstructions.
    signature = [('geometry-v2', 115000)]
    for path in (source, poses):
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256(repr(signature).encode()).hexdigest()[:16]
    return root / 'example-cache' / f'{key}-{kind}-{digest}.json'


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save_metadata(folder, metadata):
    path = folder / 'job.json'
    temporary = folder / 'job.tmp'
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    temporary.replace(path)


class Workbench:
    def __init__(self, data_dir):
        self.root = Path(data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reconstruction')
        self.processes = {}
        for folder in self.root.iterdir():
            if (folder / 'job.json').exists():
                state = read_json(folder / 'state.json', {})
                if state.get('status') in {'queued', 'running'}:
                    write_state(folder, status='interrupted', message='The server stopped during this run. Retry this method to restart.')

    def folder(self, job_id):
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise ValueError('Unknown run')
        folder = self.root / job_id
        if not (folder / 'job.json').is_file() or read_json(folder/'job.json').get('method','ours') not in (*METHOD_NAMES,'all'):
            raise ValueError('Unknown run')
        return folder

    def describe(self, folder):
        metadata = read_json(folder / 'job.json')
        state = read_json(folder / 'state.json', {})
        return dict(metadata, **legacy_progress(folder, metadata, state))

    def examples(self):
        hidden = read_json(self.root / 'hidden-examples.json', [])
        return [dict(id=key, name=value['name']) for key,value in EXAMPLES.items()
                if key not in hidden and (ROOT / 'results/mesh' / value['folder'] / 'mesh_light.ply').exists()]

    def delete_example(self, key):
        if key not in EXAMPLES:
            raise ValueError('Unknown saved reconstruction')
        with self.lock:
            path = self.root / 'hidden-examples.json'
            hidden = set(read_json(path, []))
            hidden.add(key)
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(sorted(hidden)))
            temporary.replace(path)
        return dict(deleted=True, id=key)

    def delete(self, job_id):
        with self.lock:
            folder = self.folder(job_id)
            self.cancel(job_id)
            process = self.processes.get(job_id)
        if process:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        with self.lock:
            trash = self.root / '.trash'
            trash.mkdir(exist_ok=True)
            folder.rename(trash / f'{job_id}-{time.time_ns()}')
        return dict(deleted=True, id=job_id)

    def create(self, settings):
        mesh_quality(settings)
        methods=method_settings(settings)
        name = str(settings.get('name', 'Untitled reconstruction')).strip()[:80] or 'Untitled reconstruction'
        size = settings.get('max_side', 1200)
        focal = settings.get('focal_px')
        if size not in (800, 1200, 1600, 2400):
            raise ValueError('Choose 800, 1200, 1600 or 2400 pixels')
        if focal is not None and (isinstance(focal, bool) or not isinstance(focal, (int, float)) or not 50 <= focal <= 100000):
            raise ValueError('Focal length must be between 50 and 100,000 pixels')
        if not isinstance(settings.get('make_mesh', True), bool):
            raise ValueError('Invalid output selection')
        job_id = uuid.uuid4().hex
        folder = self.root / job_id
        (folder / 'uploads').mkdir(parents=True)
        metadata = dict(id=job_id, name=name, created_at=time.time(), max_side=size,
                        focal_px=focal, make_mesh=settings.get('make_mesh', True),
                        mesh_quality=settings.get('mesh_quality', 'balanced'), uploads=[])
        metadata.update(methods)
        save_metadata(folder, metadata)
        write_state(folder, status='uploading', stage='upload', progress=0, message='Waiting for images', artifacts={}, metrics={}, warnings=[])
        return self.describe(folder)

    def upload(self, job_id, index, name, data):
        from PIL import Image
        import io
        if not re.fullmatch(r'\d{1,3}', index) or not 0 <= int(index) < 200:
            raise ValueError('Upload at most 200 images')
        formats = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tif'}
        try:
            with Image.open(io.BytesIO(data)) as image:
                suffix = formats.get(image.format)
                if not suffix or image.width * image.height > 40_000_000:
                    raise ValueError('Use JPEG, PNG or TIFF images up to 40 megapixels')
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError('This file is not a supported, readable image') from exc
        with self.lock:
            folder = self.folder(job_id)
            if read_json(folder / 'state.json')['status'] != 'uploading':
                raise ValueError('This run no longer accepts uploads')
            metadata = read_json(folder / 'job.json')
            if sum(item['bytes'] for item in metadata['uploads']) + len(data) > MAX_TOTAL:
                raise ValueError('Keep each run under 1 GB')
            stem = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(name).stem)[:60] or 'image'
            stored = f'{int(index):03d}_{stem}{suffix}'
            if any(item['index'] == int(index) for item in metadata['uploads']):
                raise ValueError('This image slot is already uploaded')
            (folder / 'uploads' / stored).write_bytes(data)
            metadata['uploads'].append(dict(index=int(index), name=Path(name).name[:150], stored=stored, bytes=len(data)))
            metadata['uploads'].sort(key=lambda item: item['index'])
            save_metadata(folder, metadata)
        return dict(uploaded=len(metadata['uploads']))

    def start(self, job_id):
        with self.lock:
            folder = self.folder(job_id)
            if read_json(folder / 'state.json')['status'] != 'uploading':
                raise ValueError('This run has already started')
            if len(read_json(folder / 'job.json')['uploads']) < 3:
                raise ValueError('Upload at least three overlapping views; 12–60 are recommended')
            metadata=read_json(folder/'job.json')
            if metadata.get('method')=='all':
                import shutil
                children=[]
                for method in METHOD_NAMES:
                    child=self.create(dict(metadata,method=method,name=metadata['name']+' · '+METHOD_NAMES[method]))
                    child_folder=self.folder(child['id']);child_meta=read_json(child_folder/'job.json')
                    for photo in metadata['uploads']:
                        source=folder/'uploads'/photo['stored'];target=child_folder/'uploads'/photo['stored']
                        try:os.link(source,target)
                        except OSError:shutil.copy2(source,target)
                    child_meta.update(uploads=metadata['uploads'],comparison_id=job_id)
                    save_metadata(child_folder,child_meta);children.append(child['id'])
                metadata.update(is_comparison=True,children=children);save_metadata(folder,metadata)
                write_state(folder,status='grouped',stage='finished',message='Two method runs created',progress=0)
                for child in children:self.start(child)
                return dict(self.describe(folder),children=children)
            metadata['generation']=uuid.uuid4().hex;save_metadata(folder,metadata)
            write_state(folder, status='queued', stage='queued', progress=0, message='Queued for reconstruction')
            self.pool.submit(self.execute,folder,metadata['generation'])
        return self.describe(folder)

    def comparisons(self):
        groups=[]
        for folder in self.root.iterdir():
            metadata=read_json(folder/'job.json',{})
            if metadata.get('is_comparison'):
                children=[]
                for child in metadata['children']:
                    p=self.root/child
                    if (p/'job.json').exists() and read_json(p/'job.json').get('method','ours') in METHOD_NAMES:children.append(self.describe(p))
                if not children:continue
                groups.append(dict(id=metadata['id'],name=metadata['name'],created_at=metadata['created_at'],children=children,
                                   missing_children=len(metadata['children'])-len(children)))
        return sorted(groups,key=lambda x:x['created_at'],reverse=True)

    def retry(self,job_id,settings):
        """Keep the full previous attempt; rerun identical uploads with explicit settings."""
        import shutil
        with self.lock:
            folder=self.folder(job_id);metadata=read_json(folder/'job.json');state=read_json(folder/'state.json',{})
            if metadata.get('is_comparison') or state.get('status') not in TERMINAL:raise ValueError('Retry an individual finished method run')
            if job_id in self.processes:raise ValueError('The previous worker is still stopping; retry shortly')
            allowed={'matching_ratio','camera_matrix','focal_px','mesh_quality','depth_backend','reuse_geometry','quality_profile','max_side','camera_prior'}
            if set(settings)-allowed:raise ValueError('Retry cannot change the method or input photos')
            reuse=settings.get('reuse_geometry',False)
            if not isinstance(reuse,bool):raise ValueError('Invalid geometry reuse option')
            if reuse:
                if not metadata.get('make_mesh'):raise ValueError('This run was created without mesh output')
                if any(key in settings and settings[key]!=metadata.get(key) for key in ('matching_ratio','camera_matrix','focal_px','quality_profile','max_side','camera_prior')):
                    raise ValueError('Camera and matching settings must stay unchanged when reusing geometry')
                if not all((folder/path).exists() for path in ('input/images','input/intrinsics.yaml','reconstruction/poses.json','reconstruction/summary.json','reconstruction/points.ply')):
                    raise ValueError('No complete saved camera reconstruction is available')
            if settings.get('max_side',metadata['max_side']) not in (800,1200,1600,2400):raise ValueError('Choose a supported photo matching size')
            updated=dict(metadata,**settings);updated['reuse_geometry']=reuse;updated.update(method_settings(updated));mesh_quality(updated)
            archive=folder/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
            retained={'attempts','uploads','thumbs'}
            if reuse:
                retained.update(('reconstruction','input','input_masks','foreground.json','camera_prior.json'))
                def snapshot(source,target):
                    try:return os.link(source,target)
                    except OSError:return shutil.copy2(source,target)
                for name in retained-{'attempts','uploads','thumbs'}:
                    source=folder/name
                    if source.is_dir():shutil.copytree(source,archive/name,copy_function=snapshot)
                    elif source.exists():snapshot(source,archive/name)
            for p in list(folder.iterdir()):
                if p.name not in retained:shutil.move(str(p),archive/p.name)
            metadata=updated;metadata['prior_attempts']=metadata.get('prior_attempts',0)+1;metadata['generation']=uuid.uuid4().hex
            save_metadata(folder,metadata);write_state(folder,status='queued',stage='queued',progress=0,message='Retry queued; previous attempt retained',artifacts={},metrics={},warnings=[])
            self.pool.submit(self.execute,folder,metadata['generation'])
        return self.describe(folder)

    def execute(self, folder,generation=None):
        with self.lock:
            if not (folder / 'job.json').exists() or (folder / 'cancelled').exists():
                return
            if generation is not None and read_json(folder/'job.json').get('generation')!=generation:return
            if read_json(folder/'state.json',{}).get('status')!='queued':return
            log = (folder / 'run.log').open('ab')
            env = dict(os.environ, PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(folder / 'mpl-cache'))
            try:
                process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('worker.py')), '--job', str(folder)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            except OSError as exc:
                log.close()
                write_state(folder, status='failed', stage='finished', message=f'Could not start reconstruction: {exc}')
                return
            self.processes[folder.name] = process
        self.monitor(folder, process, log)

    def monitor(self, folder, process, log=None):
        try:
            code = process.wait()
            with self.lock:
                if not (folder / 'job.json').exists():
                    return
                state = read_json(folder / 'state.json', {})
                if (folder / 'cancelled').exists():
                    write_state(folder, status='cancelled', stage='finished', message='Run cancelled')
                elif state.get('status') not in TERMINAL:
                    write_state(folder, status='failed', stage='finished', message=f'Reconstruction process stopped (exit {code}). Open the run log for details.')
        finally:
            if log:
                log.close()
            with self.lock:
                self.processes.pop(folder.name, None)

    def cancel(self, job_id):
        with self.lock:
            folder = self.folder(job_id)
            if read_json(folder / 'state.json')['status'] in TERMINAL:
                return self.describe(folder)
            (folder / 'cancelled').touch()
            process = self.processes.get(job_id)
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            write_state(folder, status='cancelled', stage='finished', message='Run cancelled')
            return self.describe(folder)

    def shutdown(self):
        for job_id in list(self.processes):
            self.cancel(job_id)
        self.pool.shutdown(wait=True, cancel_futures=True)


def make_handler(workbench):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'ReconstructionWorkbench/1.0'

        def log_message(self, fmt, *args):
            # Poll requests are intentionally quiet; errors still reach stderr.
            if str(args[1])[:1] in {'4', '5'}:
                super().log_message(fmt, *args)

        def reply(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def file(self, path, download=False):
            if not path.is_file():
                return self.reply(404, dict(error='File not available'))
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(path.stat().st_size))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Referrer-Policy', 'no-referrer')
            if download:
                self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open('rb') as stream:
                while chunk := stream.read(256 * 1024):
                    self.wfile.write(chunk)

        def body(self, limit):
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Chunked upload is not supported')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= limit:
                raise ValueError(f'Request must be between 1 and {limit:,} bytes')
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError('Upload was interrupted')
            return data

        def dispatch(self, mutation=False):
            try:
                self.connection.settimeout(60)
                expected = f'127.0.0.1:{self.server.server_port}'
                if self.headers.get('Host') not in {expected, f'localhost:{self.server.server_port}'}:
                    return self.reply(403, dict(error='Open this app through its localhost address'))
                if mutation:
                    origin = self.headers.get('Origin')
                    if self.headers.get('X-Reconstruction-UI') != '1' or (origin and origin != 'http://' + self.headers['Host']):
                        return self.reply(403, dict(error='Use the local reconstruction interface'))
                parsed = urlsplit(self.path)
                path = unquote(parsed.path)
                parts = path.strip('/').split('/')
                query = parse_qs(parsed.query)
                if not mutation:
                    if path == '/':
                        return self.file(STATIC / 'index.html')
                    if path.startswith('/static/'):
                        target = (STATIC / path.removeprefix('/static/')).resolve()
                        if not target.is_relative_to(STATIC.resolve()):
                            raise ValueError('Invalid asset path')
                        return self.file(target)
                    if path == '/api/jobs':
                        with workbench.lock:
                            jobs = [workbench.describe(p) for p in workbench.root.iterdir() if (p / 'job.json').exists() and not read_json(p/'job.json').get('is_comparison') and read_json(p/'job.json').get('method','ours') in METHOD_NAMES]
                            examples = workbench.examples()
                        return self.reply(200, dict(jobs=sorted(jobs, key=lambda x: x['created_at'], reverse=True), examples=examples,comparisons=workbench.comparisons()))
                    if path=='/api/capabilities':
                        return self.reply(200,dict(methods=METHOD_NAMES,max_views=MAX_UPLOAD_VIEWS,comparison_methods=list(METHOD_NAMES)))
                    if len(parts)>=3 and parts[:2]==['api','comparisons']:
                        folder=workbench.folder(parts[2])
                        if not read_json(folder/'job.json').get('is_comparison'):raise ValueError('Unknown comparison')
                        from web.comparisons import build_comparison
                        report=build_comparison(folder)
                        if len(parts)==3:return self.reply(200,report)
                        if len(parts)==5 and parts[3]=='files' and parts[4] in ('comparison.json','comparison.csv','comparison.png'):
                            return self.file(folder/parts[4],download='download' in query)
                    if path == '/api/research':
                        return self.reply(200, catalog(ROOT))
                    if path.startswith('/api/research/files/'):
                        return self.file(asset_path(ROOT,path.removeprefix('/api/research/files/')),download='download' in query)
                    if len(parts) >= 3 and parts[:2] == ['api', 'jobs']:
                        folder = workbench.folder(parts[2])
                        if len(parts) == 3:
                            return self.reply(200, workbench.describe(folder))
                        if parts[3:] == ['analysis']:
                            return self.reply(200, run_analysis(folder,read_json(folder/'job.json'),read_json(folder/'state.json',{})))
                        if parts[3:]==['preview']:
                            artifacts=read_json(folder/'state.json',{}).get('artifacts',{})
                            source=(folder/artifacts.get('mesh_ply','unavailable')).resolve()
                            poses=(folder/artifacts.get('poses','unavailable')).resolve()
                            if not source.is_relative_to(folder) or not poses.is_relative_to(folder):raise ValueError('Invalid preview source')
                            if not source.is_file() or not poses.is_file():return self.reply(404,dict(error='Mesh preview is not ready'))
                            target=folder/f'mesh-preview-{source.stat().st_mtime_ns}.png'
                            with workbench.lock:
                                if not target.exists():
                                    from web.geometry import render_preview
                                    render_preview(source,poses,target)
                            return self.file(target)
                        if parts[3:] == ['log']:
                            log = folder / 'run.log'
                            if log.exists():
                                with log.open('rb') as stream:
                                    stream.seek(max(0, log.stat().st_size - 16000))
                                    tail = stream.read().decode(errors='replace')
                            else:
                                tail = ''
                            return self.reply(200, dict(text=tail))
                        if len(parts) == 5 and parts[3] == 'files':
                            artifacts = read_json(folder / 'state.json', {}).get('artifacts', {})
                            relative = artifacts.get(parts[4])
                            if not relative:
                                raise ValueError('Output not available yet')
                            target = (folder / relative).resolve()
                            if not target.is_relative_to(folder):
                                raise ValueError('Invalid output path')
                            return self.file(target, download='download' in query)
                        if len(parts) == 5 and parts[3] == 'images':
                            metadata = read_json(folder / 'job.json')
                            image = next((item for item in metadata['uploads'] if str(item['index']) == parts[4]), None)
                            if not image:
                                raise ValueError('Image not found')
                            from PIL import Image, ImageOps
                            thumb = folder / 'thumbs' / f"{image['index']}.jpg"
                            if not thumb.exists():
                                thumb.parent.mkdir(exist_ok=True)
                                with Image.open(folder / 'uploads' / image['stored']) as raw:
                                    view = ImageOps.exif_transpose(raw).convert('RGB')
                                    view.thumbnail((300, 220))
                                    view.save(thumb, quality=85)
                            return self.file(thumb)
                    if len(parts) == 4 and parts[:2] == ['api', 'examples'] and parts[2] in EXAMPLES:
                        example = EXAMPLES[parts[2]]
                        base = ROOT / 'results/mesh' / example['folder']
                        if parts[3] == 'preview':
                            return self.file(base / 'renders/colored_0.png')
                        if parts[3] == 'analysis':
                            data=run_analysis(base,dict(name=example['name']),dict(status='completed'))
                            data.update(method='COLMAP',mesh_backend='sgbm',settings={},
                                        reconstruction=read_json((ROOT/example['poses']).parent/'summary.json',{}),
                                        mesh=read_json(base/'summary.json',{}),
                                        reference='This saved 60-view reconstruction has its own measurements, separate from the benchmark scenes in Performance. No GT geometry scores are available for this object.')
                            data['configuration']=data['reconstruction'].get('config',{})
                            data['versions']=data['reconstruction'].get('versions',{})
                            return self.reply(200,data)
                        if parts[3] in {'mesh', 'cloud'}:
                            from web.geometry import export_geometry
                            kind = parts[3]
                            source = example_source(example, kind)
                            if not source.is_file():
                                return self.reply(404, dict(error='The original reconstruction file is not available on this computer'))
                            target = example_cache(workbench.root, parts[2], example, kind)
                            with workbench.lock:
                                if not target.exists():
                                    export_geometry(source, target, ROOT / example['poses'], kind)
                            return self.file(target)
                        if parts[3] == 'cloud-ply':
                            return self.file(example_source(example, 'cloud'), download=True)
                        if parts[3] in {'ply', 'obj', 'glb'}:
                            return self.file(base / ('mesh_light.' + parts[3]), download=True)
                else:
                    if self.command == 'DELETE':
                        if len(parts) == 3 and parts[:2] == ['api', 'jobs']:
                            return self.reply(200, workbench.delete(parts[2]))
                        if len(parts) == 3 and parts[:2] == ['api', 'examples']:
                            return self.reply(200, workbench.delete_example(parts[2]))
                    if self.command == 'POST' and path == '/api/jobs':
                        return self.reply(201, workbench.create(json.loads(self.body(8192))))
                    if len(parts) >= 4 and parts[:2] == ['api', 'jobs']:
                        if self.command == 'PUT' and len(parts) == 5 and parts[3] == 'images':
                            return self.reply(200, workbench.upload(parts[2], parts[4], query.get('name', ['image'])[0], self.body(MAX_IMAGE)))
                        if self.command == 'POST' and len(parts) == 4:
                            if parts[3] == 'start':
                                return self.reply(202, workbench.start(parts[2]))
                            if parts[3] == 'cancel':
                                return self.reply(200, workbench.cancel(parts[2]))
                            if parts[3] == 'retry':
                                return self.reply(202,workbench.retry(parts[2],json.loads(self.body(8192))))
                return self.reply(404, dict(error='Not found'))
            except (ValueError, KeyError, StopIteration) as exc:
                self.reply(400, dict(error=str(exc) or 'Invalid request'))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                print(f'Workbench request failed: {exc}', file=sys.stderr)
                self.reply(500, dict(error='Could not complete the request. Check the server terminal.'))

        def do_GET(self):
            self.dispatch()

        def do_POST(self):
            self.dispatch(True)

        def do_PUT(self):
            self.dispatch(True)

        def do_DELETE(self):
            self.dispatch(True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'outputs/web_ui')
    args = parser.parse_args()
    workbench = Workbench(args.data_dir)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(workbench))
        server.daemon_threads = True
    except OSError as exc:
        workbench.shutdown()
        parser.exit(1, f'Could not start the interface: {exc}. Try --port 8766.\n')
    print(f'Reconstruction workbench: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        workbench.shutdown()


if __name__ == '__main__':
    main()
