"""Export a read-only snapshot and build the GitHub Pages edition."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STATIC = HERE.parent / 'static'
SNAPSHOT = HERE / 'snapshot'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def geometry(source, target):
    data = read(source)
    fields = ('kind', 'positions', 'colors', 'indices', 'triangles', 'total_points', 'displayed_points')
    write(target, {key: data[key] for key in fields})
    return data


def archive_item(output, slug, name, label, cloud, mesh, glb, preview, metrics, analysis):
    base = output / slug
    cloud_data = geometry(cloud, base / 'cloud.json')
    mesh_data = geometry(mesh, base / 'mesh.json')
    metrics.setdefault('points', cloud_data['total_points'])
    metrics.setdefault('triangles', mesh_data['triangles'])
    copy(glb, base / 'mesh.glb')
    copy(preview, base / 'preview.png')
    write(base / 'analysis.json', analysis)
    return dict(id=slug, name=name, label=label, status='completed', metrics=metrics,
                cloud=f'data/{slug}/cloud.json', mesh=f'data/{slug}/mesh.json',
                glb=f'data/{slug}/mesh.glb', preview=f'data/{slug}/preview.png',
                analysis=f'data/{slug}/analysis.json')


def create_snapshot(output):
    sys.path.insert(0, str(ROOT / 'code'))
    from web.research import run_analysis

    history = ROOT / 'outputs/web_ui'
    items = []
    runs = []
    for folder in history.iterdir():
        if folder.name.startswith('.') or not (folder / 'state.json').is_file():
            continue
        state = read(folder / 'state.json')
        if state.get('status') == 'completed':
            runs.append((folder, read(folder / 'job.json'), state))
    runs.sort(key=lambda run: (run[1].get('created_at', ''), run[0].name))
    for i, (folder, job, state) in enumerate(runs, 1):
        artifacts = state['artifacts']
        sources = [(folder / artifacts[key]).resolve() for key in ('cloud_view', 'mesh_view', 'mesh_glb')]
        if any(not source.is_relative_to(folder.resolve()) for source in sources):
            raise ValueError('Unexpected artifact outside the run directory')
        preview = max(folder.glob('mesh-preview-*.png'), key=lambda path: path.stat().st_mtime)
        label = {'ours': 'Our SfM', 'colmap': 'COLMAP'}[job.get('method', 'ours')]
        source = run_analysis(folder, job, state)
        # Publish only summarized measurements. Raw settings, logs, source names and paths stay local.
        analysis = {key: source[key] for key in ('method', 'mesh_backend', 'reference', 'elapsed_seconds')}
        analysis['reconstruction'] = {key: source['reconstruction'].get(key) for key in (
            'median_reprojection_px', 'mean_reprojection_px', 'mean_track_length', 'runtime_seconds')}
        analysis['history'] = [{'registered': row.get('registered')} for row in source['history']]
        mesh = source['mesh']
        analysis['mesh'] = {'timing_seconds': mesh.get('timing_seconds'), 'full': {
            key: (mesh.get('full') or {}).get(key) for key in ('components', 'boundary_edges', 'watertight')}}
        metrics = {key: state['metrics'][key] for key in ('images', 'registered', 'triangles') if key in state['metrics']}
        items.append(archive_item(output, f'run-{i:02d}', f'Run {i:02d} · {label}', label,
                                  *sources, preview, metrics, analysis))

    write(output / 'gallery.json', items)


def export_research(output):
    sys.path.insert(0, str(ROOT / 'code'))
    from web.research import asset_path, catalog

    research = catalog(ROOT)
    if not research['available']:
        raise FileNotFoundError('Benchmark evidence is missing')
    for asset in research['artifacts'] + [a for e in research['experiments'] for a in e['assets']]:
        relative = Path(asset['path'])
        source = asset_path(ROOT, asset['path'])
        target = output / 'benchmarks' / relative
        if relative.name == 'summary.json':
            summary = read(source)
            scene = summary.get('scene')
            if isinstance(scene, str) and Path(scene).is_absolute():
                summary['scene'] = Path(scene).name
                write(target, summary)
            else:
                copy(source, target)
        else:
            copy(source, target)
        asset.update(url=f'data/benchmarks/{relative.as_posix()}', bytes=target.stat().st_size)
    write(output / 'research.json', research)


def replace_once(page, pattern, replacement):
    page, count = re.subn(pattern, lambda _: replacement, page, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f'Could not find gallery template section: {pattern}')
    return page


def build(output):
    data = output / 'data'
    shutil.copytree(SNAPSHOT, data)
    export_research(data)
    research_manifest = data / 'research.json'
    research_name = f'research-{hashlib.sha256(research_manifest.read_bytes()).hexdigest()[:12]}.json'
    copy(research_manifest, data / research_name)
    manifest = output / 'data/gallery.json'
    revision = hashlib.sha256(manifest.read_bytes()).hexdigest()[:12]
    manifest_name = f'gallery-{revision}.json'
    # Keep the unversioned manifest current for tabs running the previous script.
    copy(manifest, manifest.with_name(manifest_name))
    files = ('theme.js', 'style.css', 'theme.css', 'lab.css', 'studio.css', 'palette.css',
             'motion.js', 'viewer.js', 'lab.js', 'vendor/anime.umd.min.js', 'vendor/anime.LICENSE.md')
    for name in files:
        copy(STATIC / name, output / 'static' / name)
    for name in ('gallery.js', 'gallery.css'):
        copy(HERE / name, output / 'static' / name)

    page = (STATIC / 'index.html').read_text()
    page = page.replace('"/static/', '"static/').replace('href="/"', 'href="./"')
    page = page.replace('<body>', f'<body class="gallery-edition" data-research-url="data/{research_name}" '
                        f'data-gallery-url="data/{manifest_name}">')
    lab_revision = hashlib.sha256((STATIC / 'lab.js').read_bytes()).hexdigest()[:12]
    page = page.replace('src="static/lab.js"', f'src="static/lab.js?v={lab_revision}"')
    chart_style_revision = hashlib.sha256((STATIC / 'lab.css').read_bytes()).hexdigest()[:12]
    page = page.replace('href="static/lab.css"', f'href="static/lab.css?v={chart_style_revision}"')
    script_revision = hashlib.sha256((HERE / 'gallery.js').read_bytes()).hexdigest()[:12]
    page = page.replace('<script defer src="static/app.js"></script>',
                        '<link rel="stylesheet" href="static/gallery.css">\n'
                        f'  <script defer src="static/gallery.js?v={script_revision}"></script>')
    page = page.replace('Running locally', 'Read-only archive').replace('> Workspace</button>', '> Gallery</button>')
    page = page.replace('YOUR PHOTOS, ANOTHER DIMENSION', 'RECONSTRUCTION ARCHIVE')
    page = page.replace('Upload overlapping photos of an object. Reconstruct its shape, explore it in 3D, and take the results with you.',
                        'Explore saved point clouds, meshes, and benchmark results from Reconstruction Lab.')
    page = page.replace('<button id="browse-results" type="button" class="secondary">View saved reconstructions ↓</button>', '')
    page = replace_once(page, r'<aside class="upload-panel panel">.*?</aside>',
                        '<aside class="gallery-library panel" aria-label="Saved reconstructions">\n'
                        '        <h2>Saved reconstructions</h2><div id="run-library" class="run-library">Loading saved results…</div>\n'
                        '      </aside>')
    page = page.replace('LIVE WORKSPACE', 'SAVED RESULT').replace('Your next perspective', 'Reconstruction viewer')
    page = page.replace('Ready to begin', 'Archive').replace('Your photos. A new perspective.', 'Reconstruction viewer')
    page = page.replace('Upload overlapping photos to create a 3D reconstruction,<br>or explore a saved result below.',
                        'Select a saved result to explore its geometry.')
    for pattern in (
        r'<div class="progress-area".*?</div>\s*(?=<div id="metrics")',
        r'<div id="retry-controls".*?</div>\s*(?=<section id="run-analysis")',
        r'<button id="refresh-analysis".*?</button>',
        r'<details id="run-details".*?</details>',
        r'<section class="library">.*?</section>',
        r'<button id="subtab-uploads".*?</button>',
        r'<section id="uploads-panel".*?</section>',
        r'<dialog id="delete-dialog".*?</dialog>',
    ):
        page = replace_once(page, pattern, '')
    page = replace_once(page, r'<div class="research-scope notice">.*?</div>', '')
    page = replace_once(page, r'<div class="notice"><strong>Choose one method or compare both</strong>.*?</div>', '')
    page = replace_once(page, r'<footer>.*?</footer>',
                        '<footer><span>RECONSTRUCTION LAB</span><p>Read-only snapshot of saved results and benchmark evidence.</p></footer>')
    (output / 'index.html').write_text(page)
    (output / '.nojekyll').touch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('snapshot', 'build'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    (create_snapshot if args.action == 'snapshot' else build)(args.output.resolve())


if __name__ == '__main__':
    main()
