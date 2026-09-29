import numpy as np

import scipy.sparse.linalg as spla

import skfem as fem
from skfem.models.poisson import mass, laplace

from sksparse.cholmod import cholesky as spchol

from util import log_tau


def pith_grid(z_max, nz):
    z = np.linspace(0, z_max, nz)
    mesh = fem.MeshLine(z)
    basis = fem.Basis(mesh, fem.ElementLineP1())
    C = fem.asm(mass, basis)
    G = fem.asm(laplace, basis)
    return z, C, G


def radius_grid(z_max, nz, ns, R=1):
    z = np.linspace(0, z_max, nz)
    s = np.linspace(0, R * 2*np.pi, ns+1)
    zz, ss = np.meshgrid(z, s[1:], indexing="ij")
    mesh = fem.MeshTri1DG().init_tensor(z, s, periodic=[1])
    basis = fem.Basis(mesh, fem.ElementTriP1())
    C = fem.asm(mass, basis)
    G = fem.asm(laplace, basis)
    return zz, ss, C, G


class MaternFEMSPDESampler:

    def __init__(self, C, G, d, nu, ell, sigma):
        """
        FEM-based sampler for a Matérn Gaussian field using the SPDE formulation.

        Parameters
        ----------
        C : scipy.sparse.spmatrix
            FEM mass matrix. Assumed to remain unchanged after initialization.
        G : scipy.sparse.spmatrix
            FEM stiffness matrix. Assumed to remain unchanged after initialization.
        d : int
            Spatial dimension.
        nu : float
            Matérn smoothness parameter. Must satisfy
            nu + d/2 ∈ {1, 2, 3, ...}.
        ell : float
            Matérn length scale.
        sigma : float
            Marginal standard deviation.

        Notes
        -----
        The matrices C and G are shared by cloned samplers and are therefore
        treated as read-only. Modifying either matrix after initialization may
        invalidate cached quantities such as C_L, K, its factorization, and
        log(det(K)).
        """

        assert C.shape == G.shape
        assert C.shape[0] == C.shape[1]
        assert d > 0
        assert nu > 0
        assert ell  > 0
        assert sigma > 0

        alpha_float = nu + d / 2
        alpha = round(alpha_float)
        assert np.isclose(alpha_float, alpha) and alpha >= 1, (
            "alpha = nu + d/2 must be an integer >= 1"
        )

        self.n = C.shape[0]
        self.C = C.tocsc()
        self.G = G.tocsc()
        self.d = d

        self.nu = nu
        self.ell = ell
        self.kappa = 1 / self.ell
        self.sigma = sigma
        self.alpha = alpha

        self.C_L = np.asarray(C.sum(axis=1)).ravel()
        self.logdet_C_L = np.log(self.C_L).sum()

        self.K = self.kappa**2 * self.C + self.G
        self.solve_K = spla.factorized(self.K)
        self.L_K, self.p_K = spchol(self.K, lower=True)
        self.pinv_K = np.argsort(self.p_K)
        self.logdet_K = 2.0 * np.log(self.L_K.diagonal()).sum()

        self.tau = np.exp(log_tau(self.d, self.nu, self.kappa, self.sigma))


    def __call__(self, n_samples=1):
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

        if self.alpha % 2 == 0:
            xi = np.sqrt(self.C_L)[:, None] * np.random.randn(self.n, n_samples)
            w = self.solve_K(xi)
            iters = self.alpha // 2 - 1
        else:
            z = np.random.randn(self.n, n_samples)
            z = z[self.p_K, :]
            w = spla.spsolve_triangular(self.L_K.T, z, lower=False)
            w = w[self.pinv_K, :]
            iters = (self.alpha - 1) // 2

        for _ in range(iters):
            w = self.solve_K(w * self.C_L[:, None])

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
