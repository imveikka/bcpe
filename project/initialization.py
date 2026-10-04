import numpy as np
import scipy as sci
from scipy.ndimage import map_coordinates
from multiprocess import Pool


def fit_rlts(
    points,
    fit_func,
    res_func,
    sample_size=4,
    trimming_size=0.5,
    iter_outlier_ratio=0.5,
    iter_prob=0.99,
    iterations=None,
    weighted=False,
    seed=24
):
    N = len(points)
    if trimming_size < 1:
        trimming_size = int(N * trimming_size)
    if iterations is None:
        iterations=int(np.log(1 - iter_prob) / \
                       np.log(1 - (1 - iter_outlier_ratio)**sample_size))
    if seed is not None:
        np.random.seed(seed)
    permutations = np.stack([np.random.choice(N, sample_size, replace=False) 
                             for _ in range(iterations)])
    sample_set = points[permutations]
    param_set = fit_func(sample_set)
    residuals = res_func(points, param_set)
    costs, indices = get_wls(residuals) if weighted else get_lts(residuals, trimming_size)
    best = costs.argmin()
    param = param_set[best]
    inliers = indices[best]
    return param, inliers


def get_lts(residuals, trimming_size):
    res_squared = residuals**2
    indices = np.argsort(res_squared)[..., :trimming_size]
    costs = res_squared[np.arange(len(res_squared))[:, None], indices].sum(-1)
    return costs, indices


def get_wls(residuals):
    median_res = np.median(np.abs(residuals), -1, keepdims=True)
    e_star = residuals / (6 * median_res)
    weights = np.where(np.abs(e_star) < 1, (1 - e_star**2)**2, 0)
    indices, costs = [], []
    for res, w in zip(residuals, weights):
        ix = np.where(w > 0)[0]
        cost = (w * res**2).sum()
        indices.append(ix)
        costs.append(cost)
    return np.asarray(costs), indices


def fit_circle(points, eps=1e-12):
    shift = points.mean(0)
    x, y = (points - shift).T
    z = x**2 + y**2
    features = np.c_[z, x, y, np.ones_like(z)]
    M = features.T @ features / len(points)
    s = np.mean(features, axis=0)
    N = np.array([
        [8*s[0], 4*s[1], 4*s[2], 2],
        [4*s[1], 1,      0,      0],
        [4*s[2], 0,      1,      0],
        [2,      0,      0,      0]
    ])
    values, vectors = sci.linalg.eig(M, N)
    values = values.real
    vectors = vectors.real
    values_valid = np.where(values > eps, values, np.inf)
    if not np.isfinite(values_valid).any():
        raise RuntimeError("No positive eigenvalue found")
    idx = values_valid.argmin()
    A, B, C, D = vectors[:, idx]
    param = np.array([
        -B / (2*A) + shift[0],
        -C / (2*A) + shift[1],
        np.sqrt(B*B + C*C - 4*A*D) / (2*abs(A))
    ])
    return param


def get_circle_points(param, npts=100, tmin=0, tmax=2*np.pi, endpoint=True):
    a, b, r = param
    t = np.linspace(tmin, tmax, npts, endpoint=endpoint)
    x = a + r * np.cos(t)
    y = b + r * np.sin(t)
    return x, y


def fit_func_circle(sample_set):
    param_set = filter(lambda x: x is not None, map(fit_circle, sample_set))
    return np.stack(list(param_set))


def res_func_circle(points, param_set):
    x, y = points.T
    a, b, r = param_set.T
    residuals = np.sqrt((x - a[:, None])**2 + (y - b[:, None])**2) - r[:, None]
    return residuals


def _fit_layer(param):
    section, rlts_params, size_circle = param
    if rlts_params is None:
        param = fit_circle(section[:, :2])
    else:
        param, _ = fit_rlts(section[:, :2], fit_func_circle, res_func_circle, **rlts_params)
    return np.concatenate([param, [section[0, 2]]], 0)


def init_log(points, layers, shape=(1000, 359), rlts_params=None, workers=1):

    n, m = shape
    labels = np.unique(layers)
    n_orig = len(labels)
    param_by_layer = [(points[label == layers], rlts_params, m)
                      for label in labels]

    with Pool(workers) as p:
        circles = p.map(_fit_layer, param_by_layer)
    # circles = list(map(_fit_layer, param_by_layer))
    x, y, r, z = np.stack(circles).T
    i_orig = np.arange(n_orig)
    i_interp = np.linspace(0, n_orig - 1, n)
    x, y, z, r = map(lambda arr: np.interp(i_interp, i_orig, arr), (x, y, z, r))
    return np.c_[x, y, z], np.repeat(r, m).reshape(n, m)
    
