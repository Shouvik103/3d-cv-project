"""Small reproducibility helpers; no model dependencies."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def manifest(config=None):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    versions = {}
    for name in ('numpy', 'scipy', 'opencv-python', 'PyYAML', 'Pillow', 'pycolmap'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source_root = Path(__file__).resolve().parent
    fingerprint = hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        fingerprint.update(str(path.relative_to(source_root)).encode())
        fingerprint.update(path.read_bytes())
    return dict(source_sha256=fingerprint.hexdigest(), timestamp_utc=datetime.now(timezone.utc).isoformat(), git_commit=commit,
                python=platform.python_version(), platform=platform.platform(), versions=versions, config=config)


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
