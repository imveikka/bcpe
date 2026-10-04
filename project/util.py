import numpy as np
import npx
from scipy.special import gammaln


def load_pointcloud(h5f, name, scale=1, return_layers=False):
    points = np.asarray(h5f[name]["points"])
    layers = np.array([])
    if return_layers:
        _, layers = npx.unique(points[:, -1], tol=1e-3, return_inverse=True)
    return layers, points / scale


def cut_auto(layers, points, gap=2):
    labels = np.unique(layers)
    by_layer = [(layers[label == layers], points[label == layers]) for label in labels]
    layers, points = zip(*by_layer)
    densities = [len(p) for p in points]
    changes = np.diff(densities)
    a, b = changes.argmax() + gap + 1, changes.argmin() + 1 - gap
    layers = np.concatenate(layers[a: b])
    points = np.concatenate(points[a: b])
    return layers, points


def log_tau(d, nu, kappa, sigma):
    return (
        0.5 * (gammaln(nu) - gammaln(nu + d/2))
        - np.log(sigma)
        - (d / 4) * np.log(4 * np.pi)
        - nu * np.log(kappa)
    )
