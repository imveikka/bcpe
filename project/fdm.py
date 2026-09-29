from collections.abc import Iterable
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import cholesky as spchol

from util import log_tau


def build_laplacian_1d(n, step_inv, periodic=False):
    c = step_inv**2
    L = sp.diags(
        diagonals=[-c , 2*c, -c],
        offsets=[-1, 0, 1],
        shape=(n, ), format="lil"
    )
    if periodic:
        L[0, -1] = -c
        L[-1, 0] = -c
    else:
        # Homogeneous Neumann boundaries.
        L[0, 0] = -c
        L[-1, -1] = -c
    return L


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
        out = term if term is None else out + term

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

        Notes
        -----
        The created matrices are shared by cloned samplers and are therefore
        treated as read-only. Modifying either matrix after initialization may
        invalidate cached quantities, its factorization, and log(det(K)).
        """

        self.d = len(boundaries)
        alpha_float = nu + self.d / 2
        alpha = round(alpha_float)

        self.boundaries = tuple(map(tuple, (boundaries)))
        self.shape = tuple(shape)
        self.periodic = tuple(map(bool, periodic))
        self.pad = pad
        self.shape_extended = tuple(
            n if p else n + 2
            for n, p in zip(self.shape, self.periodic)
        )
        self.step_inv = np.array([
            (n - 1 if not p else n) / (b - a)
            for (a, b), n, p in zip(self.boundaries, self.shape, self.periodic)
        ])
        self.slices = tuple(
            slice(None) if p else slice(1, -1)
            for p in self.periodic
        )

        self.n = np.prod(self.shape)
        self.nu = nu
        self.ell = ell
        self.kappa = 1 / self.ell
        self.sigma = sigma
        self.alpha = alpha

        self.L = build_laplacian(self.shape, )
        self.K = self.kappa**2 * sp.eye(self.n) + self.L
        self.solve_K = spla.factorized(self.K)
        self.L_K, self.p_K = spchol(self.K, lower=True)
        self.pinv_K = np.argsort(self.p_K)
        self.logdet_K = 2.0 * np.log(self.L_K.diagonal()).sum()

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
        if self.alpha % 2 == 0:
            w = self.solve_K(z)
            iters = self.alpha // 2 - 1
        else:
            z = z[self.p_K, :]
            w = spla.spsolve_triangular(self.L_K.T, z, lower=False)
            w = w[self.pinv_K, :]
            iters = (self.alpha - 1) // 2

        for _ in range(iters):
            w = self.solve_K(w)

        w = w.reshape((self.n, *self.shape))
        if n_samples == 1:
            w = w.ravel()

        return w.T / self.tau


    def logpdf(self, x):
        """
        Evaluate the log-density of the Matérn field.

        Parameters
        ----------
        x : ndarray, shape (n,) or (n_samples, n)
            Field realization(s).

        Returns
        -------
        float or ndarray
            Log-density for each realization.
        """

        x = np.asarray(x)
        if x.ndim == 1:
            assert x.shape[0] == self.n
            x = x[None, :]
            scalar = True
        elif x.ndim == 2:
            assert x.shape[1] == self.n
            scalar = False
        else:
            raise ValueError("x must have shape (n,) or (n_samples, n)")
        x = x.T

        qx = self.K @ x
        for _ in range(self.alpha - 1):
            qx = self.K @ (qx / self.C_L[:, None])

        quad = self.tau**2 * np.sum(x * qx, axis=0)

        logdet_Q = (
            2.0 * self.n * np.log(self.tau)
            + self.alpha * self.logdet_K
            - (self.alpha - 1) * self.logdet_C_L
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
        This is a soft copy rather than a deep copy. The FEM matrices C and G,
        and their derived mass-lumping quantities, are intentionally shared
        between the original and cloned samplers.

        If ell is unchanged, the expensive K-dependent state is also shared:
        K, its Cholesky factorization, permutation, solver, and log-determinant.
        If ell changes, these quantities are recomputed for the new K.

        The shared matrices and cached arrays are therefore assumed to be
        read-only after construction.
        """

        new = object.__new__(type(self))

        new.nu = self.nu if nu is None else nu
        new.ell = self.ell if ell is None else ell
        new.sigma = self.sigma if sigma is None else sigma

        assert new.nu > 0
        assert new.ell > 0
        assert new.sigma > 0

        new.kappa = 1 / new.ell
        new.C = self.C
        new.G = self.G
        new.d = self.d
        new.n = self.n
        new.C_L = self.C_L
        new.logdet_C_L = self.logdet_C_L

        new.tau = np.exp(log_tau(new.d, new.nu, new.kappa, new.sigma))

        alpha_float = new.nu + new.d / 2
        alpha = round(alpha_float)
        assert np.isclose(alpha_float, alpha) and alpha >= 1, (
            "alpha = nu + d/2 must be an integer >= 1"
        )
        new.alpha = alpha

        if ell is None:
            new.K = self.K
            new.L_K = self.L_K
            new.p_K = self.p_K
            new.pinv_K = self.pinv_K
            new.solve_K = self.solve_K
            new.logdet_K = self.logdet_K
        else:
            new.K = new.kappa**2 * new.C + new.G
            new.L_K, new.p_K = spchol(new.K, lower=True)
            new.pinv_K = np.argsort(new.p_K)
            new.solve_K = spla.factorized(new.K)
            new.logdet_K = 2.0 * np.log(new.L_K.diagonal()).sum()
    
        return new
