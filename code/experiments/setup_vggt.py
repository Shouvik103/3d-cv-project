"""Isolate VGGT's NumPy<2/OpenCV dependencies from the main reconstruction UI."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import sysconfig
import venv

ROOT=Path(__file__).resolve().parents[2]


def setup(target):
    target=Path(target).resolve()
    base=Path(sysconfig.get_path('purelib'))
    if not (base/'torch').is_dir():raise RuntimeError('Run this helper with the validated project .venv Python')
    if not (target/'bin/python').exists():venv.EnvBuilder(with_pip=True).create(target)
    python=target/'bin/python'
    site=Path(subprocess.check_output([str(python),'-c','import sysconfig; print(sysconfig.get_path("purelib"))'],text=True).strip())
    # This environment-only path is generated at setup time, never a hardcoded
    # source path or something distributed in the submission archive.
    (site/'project_runtime.pth').write_text(str(base)+'\n')
    subprocess.run([str(python),'-m','pip','install','--no-deps','numpy==1.26.4','opencv-python==4.11.0.86'],check=True)
    # The inherited UI stack includes pandas 3 (NumPy>=2). Give only this
    # isolated environment the compatible pandas line as well.
    subprocess.run([str(python),'-m','pip','install','pandas==2.2.3'],check=True)
    subprocess.run([str(python),'-m','pip','install','--no-deps','--ignore-installed',
                    'vggt @ git+https://github.com/facebookresearch/vggt.git@a288dd0f14786c93483e45524328726ab7b1b4ce'],check=True)
    check=subprocess.run([str(python),'-m','pip','check'],text=True,capture_output=True)
    version=subprocess.check_output([str(python),'-c','import numpy,cv2,torch,vggt;print(numpy.__version__,cv2.__version__,torch.__version__)'],text=True).strip()
    (target/'runtime_setup.json').write_text(json.dumps(dict(versions=version,pip_check=check.stdout+check.stderr,pip_check_returncode=check.returncode),indent=2))
    print('Isolated VGGT interpreter:',python,flush=True)
    print(version,flush=True)
    print(check.stdout+check.stderr,flush=True)
    if check.returncode:raise RuntimeError('Isolated VGGT dependencies are inconsistent; see runtime_setup.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'outputs/vggt_env')
    setup(parser.parse_args().out)
