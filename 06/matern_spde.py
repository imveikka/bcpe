import numpy as np
import scipy as sci
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky
from collections.abc import Iterable


def log_tau(d, nu, kappa, sigma):
    return (
        0.5 * (sci.special.gammaln(nu) - sci.special.gammaln(nu + d/2))
        - nu * np.log(kappa)
        - (d / 4) * np.log(4 * np.pi)
        - np.log(sigma)
    )


def build_laplacian_1d(n, step_inv, periodic=False):
    c = step_inv**2
    L = sp.diags(
        diagonals=[c , -2*c, c],
        offsets=[-1, 0, 1],
        shape=(n, n), format="lil"
    )
    if periodic:
        L[0, -1] = c
        L[-1, 0] = c
    else:
        # Homogeneous Neumann boundaries.
        L[0, 0] = -c
        L[-1, -1] = -c
    return L.tocsc()


def build_laplacian(shape, step_inv, periodic=None):

    assert isinstance(shape, Iterable)
    assert isinstance(step_inv, Iterable)
    assert len(shape) == len(step_inv)
    if periodic is None:
        periodic = (False,) * len(shape)
    assert isinstance(periodic, Iterable)
    assert len(periodic) == len(shape)

    out = None
    for i, (n, h_inv, p) in enumerate(zip(shape, step_inv, periodic)):
        L = build_laplacian_1d(n, h_inv, p)
        left = int(np.prod(shape[:i]))
        right = int(np.prod(shape[i+1:]))
        term = sp.kron(
            sp.kron(sp.eye(left, format="csc"), L),
            sp.eye(right, format="csc")
        )
        out = term if out is None else out + term

    return out


class MaternFDMSPDESampler:

    def __init__(self, boundaries, shape, periodic, nu, ell, sigma, pad=1):
        """
        FDM-based sampler for a Matérn Gaussian field using the SPDE formulation.

        Parameters
        ----------
        boundaries : tuple of pair of floats
            Boundaries of the regular grid
        shape : tuple
            Discretization size of each axis
        periodic : tuple
            Tuple of booleans indicating wether the grid is periodic or not
        nu : float
            Matérn smoothness parameter. Must satisfy
            nu + d/2 ∈ {1, 2, 3, ...}.
        ell : float
            Matérn length scale.
        sigma : float
            Marginal standard deviation.
        pas : int
            Size of extension beyond boundary, if axis is non-periodic

        Notes
        -----
        The created matrices are shared by cloned samplers and are therefore
        treated as read-only. Modifying either matrix after initialization may
        invalidate cached quantities, its factorization, and log(det(K)).
        """

        self.d = len(boundaries)
        alpha_float = nu + self.d / 2
        alpha = round(alpha_float)

        self.boundaries = tuple(map(tuple, boundaries))
        self.shape = tuple(shape)
        self.periodic = tuple(map(bool, periodic))
        self.pad = pad
        self.shape_extended = tuple(
            n if p else n + 2 * self.pad
            for n, p in zip(self.shape, self.periodic)
        )
        self.step_inv = np.array([
            (n - 1 if not p else n) / (b - a)
            for (a, b), n, p in zip(self.boundaries, self.shape, self.periodic)
        ])
        self.slices = tuple(
            slice(None) if p else slice(pad, -pad) if pad > 0 else slice(None)
            for p in self.periodic
        )

        self.n = np.prod(self.shape_extended)
        self.vol_sqrt = np.sqrt(np.prod(1.0 / self.step_inv))
        self.nu = nu
        self.ell = ell
        self.sigma = sigma
        self.kappa = np.sqrt(2 * self.nu) / self.ell
        self.alpha = alpha

        self.L = build_laplacian(self.shape_extended, self.step_inv, self.periodic)
        self.A = self.kappa**2 * sp.eye(self.n, format="csc") - self.L
        self.solve_A = sp.linalg.factorized(self.A)
        self.L_A, self.p_A = cholesky(self.A, lower=True)
        self.pinv_A = np.argsort(self.p_A)
        self.logdet_A = 2.0 * np.log(self.L_A.diagonal()).sum()

        self.tau = np.exp(log_tau(self.d, self.nu, self.kappa, self.sigma))


    def __call__(self, n_samples=1, extended=False):
        """
        Samples Matérn field(s).

        Parameters
        ----------
        n_samples : 
            Number of field realization(s).

        Returns
        -------
        ndarray, shape (n,), if n_samples==1, otherwise (n_samples, n)
        """

        z = np.random.randn(self.n, n_samples)
        if self.alpha % 2 == 1:
            z = z[self.p_A, :]
            w = sp.linalg.spsolve_triangular(self.L_A.T, z, lower=False)
            w = w[self.pinv_A, :]
        else:
            w = z

        for _ in range(self.alpha // 2):
            w = self.solve_A(w)

        w = w.T.reshape((n_samples, *self.shape_extended))
        if not extended:
            w = w[(slice(None), *self.slices)]
        if n_samples == 1:
            w = w[0]

        return w / (self.tau * self.vol_sqrt)


    def logpdf(self, x):
        """
        Evaluate the log-density of the Matérn field.

        Parameters
        ----------
        x : ndarray, of shape_extended or (n_samples, *shape_extended)
            Field realization(s).

        Returns
        -------
        float or ndarray
            Log-density for each realization.
        """

        if x.ndim == len(self.shape):
            assert x.shape == self.shape_extended
            x = x.ravel().reshape((1, -1))
            scalar = True
        elif x.ndim == len(self.shape) + 1:
            assert x.shape[1:] == self.shape_extended
            x = x.reshape((x.shape[0], -1))
            scalar = False
        else:
            raise ValueError("x must have shape shape_extended or (n_samples, *shape_extended)")
        x = x.T

        qx = x
        for _ in range(self.alpha):
            qx = self.A @ qx

        quad = self.tau**2 * self.vol_sqrt**2 * np.sum(x * qx, axis=0)

        logdet_Q = (
            2.0 * self.n * (np.log(self.tau) + np.log(self.vol_sqrt))
            + self.alpha * self.logdet_A
        )

        logp = (
            0.5 * logdet_Q - 0.5 * quad
            - 0.5 * self.n * np.log(2.0 * np.pi)
        )

        return logp[0] if scalar else logp



    def clone(self, *, nu=None, ell=None, sigma=None):
        """
        Create a soft copy of the sampler with optionally modified parameters.

        Parameters
        ----------
        nu : float or None, optional
        ell : float or None, optional
        sigma : float or None, optional

        If a parameter is None, the current value from the original class is retained.

        Returns
        -------
        MaternSPDESampler
            A new sampler instance with the requested parameter values.

        Notes
        -----
        This is a soft copy rather than a deep copy. matrices 
        are intentionally shared between the original and cloned samplers.

        If ell is unchanged, the expensive A-dependent state is also shared:
        K, its Cholesky factorization, permutation, solver, and log-determinant.
        If ell changes, these quantities are recomputed for the new A.

        The shared matrices and cached arrays are therefore assumed to be
        read-only after construction.
        """

        new = object.__new__(type(self))

        new.d = self.d
        new.boundaries = self.boundaries
        new.shape = self.shape
        new.periodic = self.periodic
        new.pad = self.pad
        new.shape_extended = self.shape_extended
        new.step_inv = self.step_inv
        new.slices = self.slices

        new.nu = self.nu if nu is None else nu
        new.ell = self.ell if ell is None else ell
        new.sigma = self.sigma if sigma is None else sigma
        new.kappa = np.sqrt(2 * new.nu) / new.ell
        new.tau = np.exp(log_tau(new.d, new.nu, new.kappa, new.sigma))

        new.n = self.n 
        new.vol_sqrt = self.vol_sqrt
        new.L = self.L

        alpha_float = new.nu + new.d / 2
        alpha = round(alpha_float)
        assert np.isclose(alpha_float, alpha) and alpha >= 1, (
            "alpha = nu + d/2 must be an integer >= 1"
        )
        new.alpha = alpha

        if ell is None:
            new.A = self.A
            new.L_A = self.L_A
            new.p_A = self.p_A
            new.pinv_A = self.pinv_A
            new.solve_A = self.solve_A
            new.logdet_A = self.logdet_A
        else:
            new.A = new.kappa**2 * sp.eye(new.n, format="csc") - new.L
            new.solve_A = sp.linalg.factorized(new.A)
            new.L_A, new.p_A = cholesky(new.A, lower=True)
            new.pinv_A = np.argsort(new.p_A)
            new.logdet_A = 2.0 * np.log(new.L_A.diagonal()).sum()
        return new
