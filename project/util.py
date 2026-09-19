import numpy as np
import npx


def load_pointcloud(h5f, name, scale=1, return_layers=False):
    points = np.asarray(h5f[name]["points"])
    layers = np.array([])
    if return_layers:
        _, layers = npx.unique(points[:, -1], tol=1e-3, return_inverse=True)
    return layers, points / scale