import numpy as np


def parallel_transport(xy_prime):
    """
    Parameters
    ----------
    xy_prime: (n, 2) ndarray
        xy_prime[i, 0] = x'(z_i)
        xy_primeyp[i, 1] = y'(z_i)

    Returns
    -------
        e1, e2, T
        each are (n, 3) ndarrays
    """
    N = len(xy_prime)
    tangents = np.pad(xy_prime, ((0, 0), (0, 1)), constant_values=1.0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    e1 = np.zeros((N, 3))
    e2 = np.zeros((N, 3))

    # Initial frame
    a = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(a, tangents[0])) > 0.9:
        a = np.array([1.0, 0.0, 0.0])

    e1[0] = a - np.dot(a, tangents[0]) * tangents[0]
    e1[0] /= np.linalg.norm(e1[0])

    e2[0] = np.cross(tangents[0], e1[0])
    e2[0] /= np.linalg.norm(e2[0])

    for i in range(N - 1):

        t0 = tangents[i]
        t1 = tangents[i + 1]

        axis = np.cross(t0, t1)
        s = np.linalg.norm(axis)
        d = np.clip(np.dot(t0, t1), -1.0, 1.0)

        # Tangents essentially identical
        if s < 1e-12:
            R = np.eye(3)

        else:
            axis /= s
            # Rodrigues rotation
            K = np.array([
                [0,         -axis[2],  axis[1]],
                [axis[2],   0,        -axis[0]],
                [-axis[1],  axis[0],  0]
            ])
            theta = np.arctan2(s, d)
            R = (
                np.eye(3)
                + np.sin(theta) * K
                + (1 - np.cos(theta)) * (K @ K)
            )

        e1[i + 1] = R @ e1[i]
        e2[i + 1] = R @ e2[i]

    return e1, e2, tangents


def get_pith(xy_prime, p0=np.zeros(3), dz=1.0):
    xyz_diff = np.pad(xy_prime, ((0, 0), (0, 1)), constant_values=1.0)
    return np.r_[[p0], p0 + np.cumsum(xyz_diff[:-1], axis=0) * dz]

   
def sample_log(zz, ss, sampler_pith, sampler_radius, R=1):

    z = zz[:, 0]
    dz = z[1] - z[0]
    nz, ns = zz.shape

    assert sampler_pith.shape[0] == nz
    assert sampler_radius.shape == zz.shape

    xy_prime = sampler_pith(2).T
    r = R * np.exp(sampler_radius()).reshape(nz, ns)

    c = get_pith(xy_prime, dz=dz)
    e1, e2, _ = parallel_transport(xy_prime)
    
    ez = (
        np.expand_dims(np.cos(ss / R), -1) * np.expand_dims(e1, 1) +
        np.expand_dims(np.sin(ss / R), -1) * np.expand_dims(e2, 1)
    )
    
    return c, r, ez