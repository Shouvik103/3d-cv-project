"""Portable point-cloud/camera preview and optional orbit MP4 (no OpenGL)."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sfm.io_utils import load_ply, load_poses_json
from eval.metrics import centers


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--reconstruction', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--video', action='store_true'); p.add_argument('--frames', type=int, default=180)
    a = p.parse_args(); X, rgb = load_ply(a.reconstruction/'points.ply')
    poses = load_poses_json(a.reconstruction/'poses.json'); C = centers(poses, poses['registered'])
    if len(X) > 15000:
        keep = np.random.default_rng(0).choice(len(X), 15000, replace=False); X, rgb = X[keep], rgb[keep]
    if len(X) == 0: raise ValueError('Cannot render an empty reconstruction')
    fig = plt.figure(figsize=(10, 7), facecolor='#101725'); ax = fig.add_subplot(projection='3d', facecolor='#101725')
    ax.scatter(*X.T, c=rgb/255, s=2, alpha=.9)
    ax.scatter(*C.T, c='#66d9ef', marker='^', s=35, label='Cameras')
    lo, hi = np.quantile(X, [.01, .99], axis=0); mid = (lo+hi)/2; radius = max(hi-lo)*.65
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid): setter(c-radius, c+radius)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title(f'{a.reconstruction.name} | {len(poses["registered"])} cameras | {len(X):,} points', color='white', pad=5)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.video:
        import imageio.v2 as imageio
        with imageio.get_writer(a.out, fps=30, codec='libx264', macro_block_size=2) as writer:
            for index, azimuth in enumerate(np.linspace(0, 360, a.frames, endpoint=False)):
                ax.view_init(elev=25, azim=azimuth); fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
                if index % 30 == 0: print(f'Rendered {index}/{a.frames}', flush=True)
    else:
        ax.view_init(elev=-65, azim=-90); fig.savefig(a.out, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__': main()
