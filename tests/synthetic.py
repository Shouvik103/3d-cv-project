import numpy as np


def make_scene(n_points=100, n_cameras=6, noise=0., seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n_points, 3))
    K = np.array([[700., 0, 400], [0, 700, 300], [0, 0, 1]])
    Rs, ts, obs = [], [], []
    for a in np.linspace(-0.4, 0.4, n_cameras):
        C = np.array([4*np.sin(a), .3*np.cos(2*a), -4*np.cos(a)])
        z = -C/np.linalg.norm(C); x = np.cross([0., 1., 0.], z); x /= np.linalg.norm(x)
        y = np.cross(z, x); R = np.stack([x, y, z]); t = -R @ C
        p = (K @ (R @ X.T + t[:, None])).T
        obs.append(p[:, :2]/p[:, 2:] + rng.normal(0, noise, (n_points, 2)))
        Rs.append(R); ts.append(t)
    return K, np.array(Rs), np.array(ts), X, np.array(obs)
