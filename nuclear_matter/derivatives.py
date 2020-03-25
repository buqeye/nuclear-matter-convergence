import gptools
import numpy as np
from sympy import symbols, diff, lambdify
from findiff import FinDiff
from scipy import stats
from .matter import fermi_momentum


class CustomKernel(gptools.Kernel):
    """A Custom GPTools kernel that wraps an arbitrary function f with a compatible signature

    Parameters
    ----------
    f : callable
        A positive semidefinite kernel function that takes f(Xi, Xj, ni, nj) where ni and nj are
        integers for the number of derivatives to take with respect to Xi or Xj. It should return
        an array of Xi.shape[0]
    transform : callable
        The transformation to apply to X before passing it to f. Derivatives will not take this into account via
        the chain rule.
    *args
        Args passed to the Kernel class
    **kwargs
        Kwargs passed to the Kernel class
    """

    def __init__(self, f, transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f
        self.transform = transform

    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        #         return self.f(Xi, Xj, int(np.unique(ni)[0]), int(np.unique(nj)[0]))
        dmasks = {}
        coverage = np.zeros(ni.shape[0], dtype=bool)
        n_derivs = 2
        for i in range(n_derivs + 1):
            for j in range(n_derivs + 1):
                mask_ij = ((ni == i) & (nj == j)).ravel()
                dmasks[i, j] = mask_ij
                coverage = coverage | mask_ij

        if np.any(~coverage):
            raise ValueError(f'Only up to {n_derivs} derivatives per x allowed')

        if self.transform is not None:
            Xi = self.transform(Xi)
            Xj = self.transform(Xj)

        value = np.NaN * np.ones(Xi.shape[0])
        for (i, j), mask in dmasks.items():
            value[mask] = self.f(Xi[mask], Xj[mask], i, j)
        return value


class SymmetryEnergyKernel(gptools.Kernel):

    def __init__(self, kernel_n, kernel_s, kernel_ns, kernel_sn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_n = kernel_n
        self.kernel_s = kernel_s
        self.kernel_ns = kernel_ns
        self.kernel_sn = kernel_sn
        # self.kernel_ns_c = kernel_ns_c
        # self.kernel_ns_trunc = kernel_ns_trunc

    def __call__(self, Xi, Xj, ni, nj, hyper_deriv=None, symmetric=False):
        # Assume density is passed in as X
        Kf_n_i = fermi_momentum(Xi, degeneracy=2)
        Kf_n_j = fermi_momentum(Xj, degeneracy=2)

        Kf_s_i = fermi_momentum(Xi, degeneracy=4)
        Kf_s_j = fermi_momentum(Xj, degeneracy=4)

        # kf_conversion = 2 ** (1 / 3.)  # multiply by kf_s to get kf_n. Divide kf_n by this to get kf_s

        # Assume the length scale has been modified such that all densities are neutron matter version
        # Kf_s_i = Kf_n_i
        # Kf_s_j = Kf_n_j

        if np.any(ni > 1) or np.any(nj > 1):
            raise ValueError('SymmetryEnergyKernel can currently only one derivative wrt density')
        from .matter import kf_derivative_wrt_density

        # Take derivatives with respect to density, not kf
        dKf_n_i = kf_derivative_wrt_density(Kf_n_i, Xi)
        dKf_n_j = kf_derivative_wrt_density(Kf_n_j, Xj)
        dKf_s_i = kf_derivative_wrt_density(Kf_s_i, Xi)
        dKf_s_j = kf_derivative_wrt_density(Kf_s_j, Xj)

        factor_n_i = np.ones(Xi.shape[0])
        factor_s_i = np.ones(Xi.shape[0])
        factor_n_i[ni.ravel() == 1] = dKf_n_i[ni.ravel() == 1].ravel()
        factor_s_i[ni.ravel() == 1] = dKf_s_i[ni.ravel() == 1].ravel()

        factor_n_j = np.ones(Xj.shape[0])
        factor_s_j = np.ones(Xj.shape[0])
        factor_n_j[nj.ravel() == 1] = dKf_n_j[nj.ravel() == 1].ravel()
        factor_s_j[nj.ravel() == 1] = dKf_s_j[nj.ravel() == 1].ravel()

        cov_n = self.kernel_n(Kf_n_i, Kf_n_j, ni=ni, nj=nj, hyper_deriv=hyper_deriv, symmetric=symmetric)
        cov_s = self.kernel_s(Kf_s_i, Kf_s_j, ni=ni, nj=nj, hyper_deriv=hyper_deriv, symmetric=symmetric)
        # Assumes this takes Kf_n
        cov_ns = self.kernel_ns(Kf_n_i, Kf_n_j, ni=ni, nj=nj, hyper_deriv=hyper_deriv, symmetric=False)
        cov_sn = self.kernel_sn(Kf_n_i, Kf_n_j, ni=ni, nj=nj, hyper_deriv=hyper_deriv, symmetric=False)
        # cov_ns_c = self.kernel_ns_c(Kf_n_i, Kf_n_j, ni=ni, nj=nj, hyper_deriv=hyper_deriv, symmetric=symmetric)
        # cov_sn_c = self.kernel_ns_c(Kf_n_i, Kf_n_j, ni=ni, nj=nj, hyper_deriv=hyper_deriv, symmetric=symmetric)
        # cov_sn_c = cov_ns_c.copy()

        # print(cov_n.shape)

        cov_n *= factor_n_i * factor_n_j
        cov_s *= factor_s_i * factor_s_j
        # cov_ns *= factor_n_i * factor_s_j
        # cov_sn *= factor_s_i * factor_n_j
        cov_ns *= factor_n_i * factor_n_j
        cov_sn *= factor_n_i * factor_n_j
        return cov_n + cov_s - cov_ns - cov_sn


#         print(ni, nj, flush=True)
# Only works for 1d X.
#         return np.array([
#             self.f(x_i, x_j, int(n_i[0]), int(n_j[0]))
#             for x_i, x_j, n_i, n_j in zip(Xi, Xj, ni, nj)
#         ])


def kernel_scale_sympy(
        lowest_order=4, highest_order=None, include_3bf=False, k_f1_scale=1., k_f2_scale=1., off_diag=False
):
    """Creates a sympy object that is the convergence part of the GP kernel

    Parameters
    ----------
    lowest_order
    highest_order
    include_3bf
    k_f1_scale
    k_f2_scale
    off_diag
    """
    from sympy import sqrt
    k_f1_orig, k_f2_orig, y_ref, Lambda_b, = symbols('k_f1 k_f2 y_ref Lambda_b')
    k_f1 = k_f1_orig * k_f1_scale
    k_f2 = k_f2_orig * k_f2_scale
    Q1 = k_f1 / Lambda_b
    Q2 = k_f2 / Lambda_b

    if off_diag:
        # Take into account coregional kernel which can correlate two processes
        num1 = Q1 ** (2 * lowest_order)
        num2 = Q2 ** (2 * lowest_order)
        if highest_order is not None:
            num1 = num1 - Q1 ** (2 * (highest_order + 1))
            num2 = num2 - Q2 ** (2 * (highest_order + 1))
        frac1 = (num1 / (1 - Q1 ** 2)) ** 0.5
        frac2 = (num2 / (1 - Q2 ** 2)) ** 0.5
        kernel_scale = y_ref ** 2 * frac1 * frac2
    else:
        # The standard use case
        num = (Q1 * Q2) ** lowest_order
        if highest_order is not None:
            num = num - (Q1 * Q2) ** (highest_order + 1)
        kernel_scale = y_ref ** 2 * num / (1 - Q1 * Q2)
    kernel_scale *= (k_f1 * k_f2)**2  # Make y_ref have a quadratic piece

    if include_3bf and (highest_order is None or highest_order >= 3):
        lowest_order_3bf = lowest_order if lowest_order >= 3 else 3
        num_3bf = (Q1 * Q2) ** lowest_order_3bf
        if highest_order is not None:
            num_3bf = num_3bf - (Q1 * Q2) ** (highest_order + 1)

        # kf_3bf_order = 3
        kf_3bf_order = 1  # Actually, linear doesn't look so bad
        kernel_scale += (k_f1 * k_f2)**kf_3bf_order * y_ref ** 2 * num_3bf / (1 - Q1 * Q2)

    return k_f1_orig, k_f2_orig, Lambda_b, y_ref, kernel_scale


def eval_kernel_scale(Xi, Xj=None, ni=None, nj=None, breakdown=600, ref=16, lowest_order=4,
                      highest_order=None, include_3bf=False):
    """Creates a matrix for the convergence part of the GP kernel.
    Compatible with the CustomKernel class signature.

    Parameters
    -----------
    Xi
    Xj
    ni
    nj
    breakdown
    ref
    lowest_order
    highest_order
    include_3bf
    """
    if ni is None:
        ni = 0
    if nj is None:
        nj = 0
    k_f1, k_f2, Lambda_b, y_ref, kernel_scale = kernel_scale_sympy(
        lowest_order=lowest_order, highest_order=highest_order, include_3bf=include_3bf
    )
    expr = diff(kernel_scale, k_f1, ni, k_f2, nj)
    f = lambdify((k_f1, k_f2, Lambda_b, y_ref), expr, "numpy")
    if Xj is None:
        Xj = Xi
    K = f(Xi, Xj, breakdown, ref)
    K = K.astype('float')
    return np.squeeze(K)


class ConvergenceKernel:

    def __init__(
            self, breakdown=600, ref=16, lowest_order=4, highest_order=None, include_3bf=False,
            k_f1_scale=1., k_f2_scale=1., off_diag=False,
    ):

        # TODO: Fix the reference scale for 3bf
        hbar_c = 197.3269718  # MeV fm
        breakdown = breakdown / hbar_c  # convert to inverse fermi to match kf
        self.breakdown = breakdown
        self.ref = ref
        self.lowest_order = lowest_order
        self.highest_order = highest_order

        k_f1, k_f2, Lambda_b, y_ref, kernel_scale = kernel_scale_sympy(
            lowest_order=lowest_order, highest_order=highest_order, include_3bf=include_3bf,
            k_f1_scale=k_f1_scale, k_f2_scale=k_f2_scale, off_diag=off_diag
        )
        self.k_f1 = k_f1
        self.k_f2 = k_f2
        self.Lambda_b = Lambda_b
        self.y_ref = y_ref
        self.kernel_scale = kernel_scale
        self._funcs = {}

        self.ni_symbol, self.nj_symbol = symbols('n_i, n_j')

    def compute_func(self, ni, nj):
        if (ni, nj) in self._funcs:
            return self._funcs[ni, nj]
        else:
            k_f1 = self.k_f1
            k_f2 = self.k_f2
            Lambda_b = self.Lambda_b
            y_ref = self.y_ref
            kernel_scale = self.kernel_scale

            expr = diff(kernel_scale, k_f1, ni, k_f2, nj)
            f = lambdify((k_f1, k_f2, Lambda_b, y_ref), expr, "numpy")
            self._funcs[ni, nj] = f
            return f

    def __call__(self, Xi, Xj=None, ni=None, nj=None):
        if ni is None:
            ni = 0
        if nj is None:
            nj = 0
        if Xj is None:
            Xj = Xi

        breakdown = self.breakdown
        ref = self.ref
        f = self.compute_func(ni, nj)
        K = f(Xi, Xj, breakdown, ref)
        #         K = f(Xi, Xj, breakdown, ref, ni, nj)
        # try:
        #     K = K.astype('float')
        # except:
        #     pass
        return np.squeeze(K)


def predict_with_derivatives(gp, X, n=0, only_cov=False, **kwargs):
    n = np.atleast_1d(n)
    X_tiled = np.concatenate([X for _ in n], axis=0)
    n_tiled = np.concatenate([n_i * np.ones(X.shape[0], dtype=int) for n_i in n])[:, None]
    if only_cov:
        return gp.compute_Kij(X_tiled, None, ni=n_tiled, nj=None)
    return gp.predict(X_tiled, n=n_tiled, **kwargs)


def extract_blocks(a, blocksize, keep_as_view=False):
    M, N = a.shape
    b0, b1 = blocksize
    if keep_as_view:
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2)
    else:
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)


def extract_means(m, length):
    return m.reshape(-1, length)


def get_blocks_map(a, blocksize):
    blocks = extract_blocks(a, blocksize, keep_as_view=True)
    d = {}
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            d[i, j] = blocks[i, j].copy()
    return d


def get_means_map(m, length):
    means = extract_means(m, length)
    d = {}
    for i, m_i in enumerate(means):
        d[i] = m_i
    return d


def get_std_map(cov_map):
    d = {}
    for (i, j), cov_ij in cov_map.items():
        if i == j:
            d[i] = np.sqrt(np.diag(cov_ij))
    return d


def compute_mean_cov_blm(X, y, Sigma_y=0, mean0=0, cov0=0):
    R"""Estimate parameters of the Bayesian Linear Model

    Parameters
    ----------
    X : array-like, shape = (N, n_params)
        The feature matrix
    y : array-like, shape = (N,)
        The data
    Sigma_y : int or array-like, shape = (N, N)
        The data covariance
    mean0 : int or array-like, shape = (n_params, n_params)
        The prior mean on the polynomial coefficients
    cov0 : int or array-like, shape = (n_params, n_params)
        The prior covariance on the polynomial coefficients. If zero, the prior is uninformative

    Returns
    -------
    mean : array-like
    cov : array-like
    """
    ones_y = np.ones(y.shape[0], dtype=float)
    ones_n_params = np.ones(X.shape[-1], dtype=float)
    Sigma_y = Sigma_y * ones_y
    mean0 = mean0 * ones_n_params
    if Sigma_y.ndim == 1:
        Sigma_y = np.diag(Sigma_y)

    if cov0 == 0:
        prec0 = 0 * np.diag(ones_n_params)
    else:
        cov0 = cov0 * ones_n_params
        if cov0.ndim == 1:
            cov0 = np.diag(cov0)
        prec0 = np.linalg.inv(cov0)

    cov = np.linalg.inv(prec0 + X.T @ np.linalg.solve(Sigma_y, X))
    mean = cov @ (prec0 @ mean0 + X.T @ np.linalg.solve(Sigma_y, y))
    return mean, cov


def compute_log_evidence_blm(X, y, Sigma_y, mean0=0, cov0=0):
    mean, cov = compute_mean_cov_blm(X, y, Sigma_y, mean0, cov0)
    mean_y = X @ mean
    cov_y = X @ cov @ X.T + Sigma_y

    logdet = np.linalg.slogdet(2 * np.pi * cov_y)[1]
    logpdf = - 0.5 * (y - mean_y) @ np.linalg.solve(cov_y, y - mean_y) - 0.5 * logdet

    # try:
    #     y_dist = stats.multivariate_normal(mean=mean_y, cov=cov_y, allow_singular=True)
    #     logpdf2 = y_dist.logpdf(y)
    #     # print(np.allclose(logpdf2, logpdf))
    #     if not np.allclose(logpdf2, logpdf):
    #         print('stats', logpdf2)
    #         print('mine', logpdf)
    #         print('diff', np.abs(logpdf - logpdf2))
    #     else:
    #         print('hooray!')
    # except (np.linalg.LinAlgError, ValueError):
    #     print('Singular, skipping..')
    return logpdf


class ObservableContainer:

    def __init__(
            self, density, kf, y, orders, density_interp, kf_interp,
            std, ls, ref, breakdown, err_y=0, derivs=(0, 1, 2), include_3bf=True, verbose=False
    ):

        self.density = density
        self.kf = kf
        self.Kf = Kf = kf[:, None]

        self.density_interp = density_interp
        self.kf_interp = kf_interp
        self.Kf_interp = Kf_interp = kf_interp[:, None]
        self.X_interp = Kf_interp

        self.y = y
        self.N_interp = N_interp = len(kf_interp)
        err_y = np.broadcast_to(err_y, y.shape[0])  # Turn to vector if not already
        self.err_y = err_y
        self.Sigma_y = np.diag(err_y**2)  # Make a diagonal covariance matrix
        self.derivs = derivs

        self.gps_interp = {}
        self.gps_trunc = {}

        self._y_interp_all_derivs = {}
        self._cov_interp_all_derivs = {}
        self._y_interp_vecs = {}
        self._std_interp_vecs = {}
        self._cov_interp_blocks = {}

        self._dy_dn = {}
        self._d2y_dn2 = {}
        self._dy_dk = {}
        self._d2y_dk2 = {}
        self._y_dict = {}

        d_dn = FinDiff(0, density, 1)
        d2_dn2 = FinDiff(0, density, 2, acc=2)
        d_dk = FinDiff(0, kf, 1)
        d2_dk2 = FinDiff(0, kf, 2, acc=2)

        self._cov_total_all_derivs = {}
        self._cov_total_blocks = {}
        self._std_total_vecs = {}

        # The priors on the interpolator parameters
        self.mean0 = 0
        self.cov0 = 0
        self._best_max_orders = {}
        self._start_poly_order = 2

        # from scipy.interpolate import splrep
        from scipy.interpolate import UnivariateSpline
        self.splines = {}

        self.coeff_kernel = gptools.SquaredExponentialKernel(
            initial_params=[std, ls], fixed_params=[True, True])
        for i, n in enumerate(orders):
            first_omitted = n + 1
            if first_omitted == 1:
                first_omitted += 1  # the Q^1 contribution is zero, so bump to Q^2
            _kern_lower = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref, lowest_order=0, highest_order=n, include_3bf=include_3bf
            ))
            kern_interp = _kern_lower * self.coeff_kernel
            _kern_upper = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref, lowest_order=first_omitted, include_3bf=include_3bf
            ))
            kern_trunc = _kern_upper * self.coeff_kernel

            # try:
            #     err_y_i = err_y[i]
            # except TypeError:
            #     err_y_i = err_y

            y_n = y[:, i]
            self._y_dict[n] = y_n

            # Interpolating processes
            # mu_n = gptools.ConstantMeanFunction(initial_params=[np.mean(y_n)])
            # mu_n = gptools.ConstantMeanFunction(initial_params=[np.max(y_n)+20])
            mu_n = gptools.ConstantMeanFunction(initial_params=[0])
            gp_interp = gptools.GaussianProcess(kern_interp, mu=mu_n)
            gp_interp.add_data(Kf, y_n, err_y=err_y)
            # gp_interp.optimize_hyperparameters(max_tries=10)  # For the mean
            self.gps_interp[n] = gp_interp

            # Finite difference:
            self._dy_dn[n] = d_dn(y_n)
            self._d2y_dn2[n] = d2_dn2(y_n)
            self._dy_dk[n] = d_dk(y_n)
            self._d2y_dk2[n] = d2_dk2(y_n)

            # Fractional interpolator polynomials
            self._best_max_orders[n] = self.compute_best_interpolator(
                density, y=y_n, start_order=self._start_poly_order, max_order=10
            )
            self.splines[n] = UnivariateSpline(density, y_n, s=np.max(err_y))
            if verbose:
                print(f'For EFT order {n}, the best polynomial has max nu = {self._best_max_orders[n]}')

            # Back to GPs:

            y_interp_all_derivs_n, cov_interp_all_derivs_n = predict_with_derivatives(
                gp=gp_interp, X=Kf_interp, n=derivs, return_cov=True
            )

            y_interp_vecs_n = get_means_map(y_interp_all_derivs_n, N_interp)
            cov_interp_blocks_n = get_blocks_map(cov_interp_all_derivs_n, (N_interp, N_interp))
            std_interp_vecs_n = get_std_map(cov_interp_blocks_n)

            self._y_interp_all_derivs[n] = y_interp_all_derivs_n
            self._cov_interp_all_derivs[n] = cov_interp_all_derivs_n
            self._y_interp_vecs[n] = y_interp_vecs_n
            self._cov_interp_blocks[n] = cov_interp_blocks_n
            self._std_interp_vecs[n] = std_interp_vecs_n

            # Truncation Processes
            gp_trunc = gptools.GaussianProcess(kern_trunc)
            self.gps_trunc[n] = gp_trunc

            cov_trunc_all_derivs_n = predict_with_derivatives(
                gp=gp_trunc, X=Kf_interp, n=derivs, only_cov=True
            )
            cov_total_all_derivs_n = cov_interp_all_derivs_n + cov_trunc_all_derivs_n

            cov_total_blocks_n = get_blocks_map(cov_total_all_derivs_n, (N_interp, N_interp))
            std_total_vecs_n = get_std_map(cov_total_blocks_n)

            self._cov_total_all_derivs[n] = cov_total_all_derivs_n
            self._cov_total_blocks[n] = cov_total_blocks_n
            self._std_total_vecs[n] = std_total_vecs_n

    def get_cov(self, order, deriv1, deriv2=None, include_trunc=True):
        if deriv2 is None:
            deriv2 = deriv1
        if include_trunc:
            covs = self._cov_total_blocks[order]
        else:
            covs = self._cov_interp_blocks[order]
        return covs[deriv1, deriv2]

    def get_deriv_cov(self, order, idx, derivs=None, include_trunc=True):
        if include_trunc:
            covs = self._cov_total_blocks[order]
        else:
            covs = self._cov_interp_blocks[order]
        if derivs is None:
            derivs = self.derivs
        cov = np.zeros((len(derivs), len(derivs)))
        for i, d1 in enumerate(derivs):
            for j, d2 in enumerate(derivs):
                cov[i, j] = covs[d1, d2][idx, idx]
        return cov

    def get_pred(self, order, deriv, spline=False):
        return self._y_interp_vecs[order][deriv]

    def spline(self, density, order, deriv):
        return self.splines[order].derivative(deriv)(density)

    def get_std(self, order, deriv, include_trunc=True):
        if include_trunc:
            std = self._std_total_vecs[order]
        else:
            std = self._std_interp_vecs[order]
        return std[deriv]

    def draw_sample(self, order, num_samp=1, include_trunc=True):
        gp = gptools.GaussianProcess(k=self.gps_trunc[order].k)  # Kernel won't matter
        mean = self._y_interp_all_derivs[order]
        if include_trunc:
            cov = self._cov_total_all_derivs[order]
        else:
            cov = self._cov_interp_all_derivs[order]
        # samples shape: n_derivs * N_interp, num_samp
        samples = gp.draw_sample(Xstar=self.X_interp, num_samp=num_samp, mean=mean, cov=cov)
        # change it to: n_derivs, N_interp, num_samp
        sample_blocks = extract_blocks(samples, blocksize=(self.N_interp, samples.shape[-1]))
        sample_dict = {}
        # self.derivs = [0, 1, 2]
        # Put into dict for access via derivative value
        for i, d in enumerate(self.derivs):
            sample_dict[d] = sample_blocks[i]
        return sample_dict

    def predict(self, X, order, derivs=None, include_trunc=True):
        """Predict from the GP

        Parameters
        ----------
        X : array
            The variable taken by the GP, which is the fermi momentum Kf.
        order
        derivs
        include_trunc

        Returns
        -------

        """
        if derivs is None:
            derivs = self.derivs
        y_interp, cov = predict_with_derivatives(
            self.gps_interp[order], X=X, n=derivs, return_cov=True
        )
        if include_trunc:
            cov += predict_with_derivatives(
                self.gps_trunc[order], X=X, n=derivs, only_cov=True
            )
        return y_interp, cov

    def finite_difference(self, order, deriv=1, wrt_kf=True):
        y = self._y_dict
        if wrt_kf:
            dy_dx = self._dy_dk
            d2y_dx2 = self._d2y_dk2
        else:
            dy_dx = self._dy_dn
            d2y_dx2 = self._d2y_dn2
        if deriv == 0:
            return y[order]
        if deriv == 1:
            return dy_dx[order]
        elif deriv == 2:
            return d2y_dx2[order]
        else:
            raise ValueError('deriv must be 0, 1 or 2')

    def poly_interp(self, order, deriv=0, density=None, wrt_kf=False):
        if density is None:
            density = self.density
        if wrt_kf:
            raise NotImplementedError('wrt_kf = True is not ready')
        poly_order = self._best_max_orders[order]
        X = self.compute_feature_matrix_fractional_interpolator(
            density, start_order=self._start_poly_order, end_order=poly_order, deriv=0
        )
        mean, _ = compute_mean_cov_blm(
            X, y=self._y_dict[order], Sigma_y=self.Sigma_y, mean0=self.mean0, cov0=self.cov0
        )
        X_deriv = self.compute_feature_matrix_fractional_interpolator(
            density, start_order=self._start_poly_order, end_order=poly_order, deriv=deriv
        )
        # print(f'Order {order}, mean {mean}')
        return X_deriv @ mean

    def compute_functional_coefficients_df(self):
        density = self.density
        functional_orders = np.arange(self._start_poly_order, max(self._best_max_orders.values()) + 1)
        coeffs = {}
        for order, poly_order in self._best_max_orders.items():
            X = self.compute_feature_matrix_fractional_interpolator(
                density, start_order=self._start_poly_order, end_order=poly_order, deriv=0
            )
            c, _ = compute_mean_cov_blm(
                X, y=self._y_dict[order], Sigma_y=self.Sigma_y, mean0=self.mean0, cov0=self.cov0
            )
            padding = np.NaN * np.ones((len(functional_orders) - len(c)))
            c = np.concatenate((c, padding))
            coeffs['EFT Order ' + str(order)] = c
        import pandas as pd

        df = pd.DataFrame.from_dict(coeffs)
        df['nu'] = functional_orders
        return df.set_index('nu')

    @staticmethod
    def compute_feature_matrix_fractional_interpolator(density, start_order=2, end_order=10, deriv=0):
        n = density
        n0 = 0.16
        fit_orders = np.arange(start_order, end_order+1)
        X = []
        for nu in fit_orders:
            factor = 1
            exponent = nu / 3
            if deriv > 0:
                for _ in range(deriv):
                    factor *= exponent / n0
                    exponent -= 1
            X.append(factor * (n / n0) ** exponent)
        return np.asarray(X).T
        # return np.asarray([(n / n0) ** (nu / 3) for nu in fit_orders]).T

    def compute_best_interpolator(self, density, y, start_order=2, max_order=10):
        fit_orders = np.arange(start_order, max_order+1)
        log_evidences = []
        for i, nu in enumerate(fit_orders):
            X_i = self.compute_feature_matrix_fractional_interpolator(density, start_order, end_order=nu, deriv=0)
            log_evidences.append(compute_log_evidence_blm(
                X_i, y=y, Sigma_y=self.Sigma_y, mean0=self.mean0, cov0=self.cov0
            ))
        log_evidences = np.array(log_evidences)
        log_evidences -= np.max(log_evidences)
        evidences = np.exp(log_evidences)
        return fit_orders[np.argmax(evidences)]


class SymmetryEnergyContainer(ObservableContainer):

    def __init__(
            self, density, y, orders, density_interp,
            std_n, ls_n, std_s, ls_s, ref_n, ref_s, breakdown, err_y=0, derivs=(0, 1, 2), include_3bf=True,
            verbose=False, rho=None
    ):
        self.density = density
        self.Density = Density = density[:, None]
        self.kf = None
        self.Kf = None

        self.density_interp = density_interp
        self.Density_interp = Density_interp = density_interp[:, None]
        self.kf_interp = None
        self.Kf_interp = None
        self.X_interp = Density_interp

        self.y = y
        self.N_interp = N_interp = len(density_interp)
        err_y = np.broadcast_to(err_y, y.shape[0])  # Turn to vector if not already
        self.err_y = err_y
        self.Sigma_y = np.diag(err_y ** 2)  # Make a diagonal covariance matrix
        self.derivs = derivs

        self.gps_interp = {}
        self.gps_trunc = {}

        self._y_interp_all_derivs = {}
        self._cov_interp_all_derivs = {}
        self._y_interp_vecs = {}
        self._std_interp_vecs = {}
        self._cov_interp_blocks = {}

        self._dy_dn = {}
        self._d2y_dn2 = {}
        self._dy_dk = {}
        self._d2y_dk2 = {}
        self._y_dict = {}

        d_dn = FinDiff(0, density, 1)
        d2_dn2 = FinDiff(0, density, 2, acc=2)
        # d_dk = FinDiff(0, kf, 1)
        # d2_dk2 = FinDiff(0, kf, 2, acc=2)

        self._cov_total_all_derivs = {}
        self._cov_total_blocks = {}
        self._std_total_vecs = {}

        # The priors on the interpolator parameters
        self.mean0 = 0
        self.cov0 = 0
        self._best_max_orders = {}
        self._start_poly_order = 2

        self.ref_n = ref_n
        self.ref_s = ref_s

        kf_conversion = 2 ** (1 / 3.)

        if rho is not None:
            ls_s = ls_n / kf_conversion
        else:
            ls_s_scaled = kf_conversion * ls_s

        from functools import partial
        # transform_n = partial(fermi_momentum, degeneracy=2)
        # transform_s = partial(fermi_momentum, degeneracy=4)

        self.coeff_kernel_n = gptools.SquaredExponentialKernel(
            initial_params=[std_n, ls_n], fixed_params=[True, True])
        # Assumes the symmetric nuclear matter kernel takes kf_s as an argument, so use ls_s
        self.coeff_kernel_s = gptools.SquaredExponentialKernel(
            initial_params=[std_s, ls_s], fixed_params=[True, True])

        if rho is not None:
            # only use ls_n, and assume rho is the correlation of the off-diagonal
            std_off = np.sqrt(std_s * std_n) * rho
            ls_off = ls_n
        else:
            # But the off-diagonal will take kf_n as an argument, so use scaled length scale
            std_off = np.sqrt(std_s * std_n) * (2 * ls_n * ls_s_scaled / (ls_n**2 + ls_s_scaled**2)) ** 0.25
            ls_off = np.sqrt((ls_s_scaled**2 + ls_n**2) / 2)
        ref_off = np.sqrt(ref_s * ref_n)
        self.coeff_kernel_off = gptools.SquaredExponentialKernel(
            initial_params=[std_off, ls_off], fixed_params=[True, True])

        print(ls_n, ls_s, ls_off)

        for i, n in enumerate(orders):
            first_omitted = n + 1
            if first_omitted == 1:
                first_omitted += 1  # the Q^1 contribution is zero, so bump to Q^2
            _kern_lower_n = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_n, lowest_order=0, highest_order=n, include_3bf=include_3bf
            ))
            _kern_lower_s = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_s, lowest_order=0, highest_order=n, include_3bf=include_3bf
            ))
            _kern_lower_ns = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_off, lowest_order=0, highest_order=n,
                include_3bf=include_3bf,
                k_f1_scale=1, k_f2_scale=1./kf_conversion,  # Will turn kf_n to kf_s
                # off_diag=True
            ))
            _kern_lower_sn = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_off, lowest_order=0, highest_order=n,
                include_3bf=include_3bf,
                k_f1_scale=1./kf_conversion, k_f2_scale=1,  # Will turn kf_n to kf_s
                # off_diag=True
            ))
            kern_interp_n = _kern_lower_n * self.coeff_kernel_n
            kern_interp_s = _kern_lower_s * self.coeff_kernel_s
            kern_interp_ns = _kern_lower_ns * self.coeff_kernel_off
            kern_interp_sn = _kern_lower_sn * self.coeff_kernel_off
            kern_interp = SymmetryEnergyKernel(
                kernel_n=kern_interp_n,
                kernel_s=kern_interp_s,
                kernel_ns=kern_interp_ns,
                kernel_sn=kern_interp_sn,
            )

            _kern_upper_n = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_n, lowest_order=first_omitted, include_3bf=include_3bf
            ))
            _kern_upper_s = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_s, lowest_order=first_omitted, include_3bf=include_3bf
            ))
            _kern_upper_ns = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_off, lowest_order=first_omitted, include_3bf=include_3bf,
                k_f1_scale=1, k_f2_scale=1/kf_conversion,
                # off_diag=True
            ))
            _kern_upper_sn = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, ref=ref_off, lowest_order=first_omitted, include_3bf=include_3bf,
                k_f1_scale=1/kf_conversion, k_f2_scale=1,
                # off_diag=True
            ))
            kern_trunc_n = _kern_upper_n * self.coeff_kernel_n
            kern_trunc_s = _kern_upper_s * self.coeff_kernel_s
            kern_trunc_ns = _kern_upper_ns * self.coeff_kernel_off
            kern_trunc_sn = _kern_upper_sn * self.coeff_kernel_off
            kern_trunc = SymmetryEnergyKernel(
                kernel_n=kern_trunc_n,
                kernel_s=kern_trunc_s,
                kernel_ns=kern_trunc_ns,
                kernel_sn=kern_trunc_sn,
            )

            y_n = y[:, i]
            self._y_dict[n] = y_n

            # Interpolating processes
            # mu_n = gptools.ConstantMeanFunction(initial_params=[np.mean(y_n)])
            # mu_n = gptools.ConstantMeanFunction(initial_params=[np.max(y_n)+20])
            mu_n = gptools.ConstantMeanFunction(initial_params=[0])
            gp_interp = gptools.GaussianProcess(kern_interp, mu=mu_n)
            gp_interp.add_data(Density, y_n, err_y=err_y)
            # gp_interp.optimize_hyperparameters(max_tries=10)  # For the mean
            self.gps_interp[n] = gp_interp

            # Finite difference:
            self._dy_dn[n] = d_dn(y_n)
            self._d2y_dn2[n] = d2_dn2(y_n)
            # self._dy_dk[n] = d_dk(y_n)
            # self._d2y_dk2[n] = d2_dk2(y_n)

            # Fractional interpolator polynomials
            self._best_max_orders[n] = self.compute_best_interpolator(
                density, y=y_n, start_order=self._start_poly_order, max_order=10
            )
            if verbose:
                print(f'For EFT order {n}, the best polynomial has max nu = {self._best_max_orders[n]}')

            # Back to GPs:

            y_interp_all_derivs_n, cov_interp_all_derivs_n = predict_with_derivatives(
                gp=gp_interp, X=Density_interp, n=derivs, return_cov=True
            )

            y_interp_vecs_n = get_means_map(y_interp_all_derivs_n, N_interp)
            cov_interp_blocks_n = get_blocks_map(cov_interp_all_derivs_n, (N_interp, N_interp))
            # for (ii, jj), cov_ij in cov_interp_blocks_n.items():
            #     cov_interp_blocks_n[ii, jj] += 1e-12 * np.eye(cov_ij.shape[0])
            std_interp_vecs_n = get_std_map(cov_interp_blocks_n)

            self._y_interp_all_derivs[n] = y_interp_all_derivs_n
            self._cov_interp_all_derivs[n] = cov_interp_all_derivs_n
            self._y_interp_vecs[n] = y_interp_vecs_n
            self._cov_interp_blocks[n] = cov_interp_blocks_n
            self._std_interp_vecs[n] = std_interp_vecs_n

            # Truncation Processes
            gp_trunc = gptools.GaussianProcess(kern_trunc)
            self.gps_trunc[n] = gp_trunc

            cov_trunc_all_derivs_n = predict_with_derivatives(
                gp=gp_trunc, X=Density_interp, n=derivs, only_cov=True
            )
            cov_total_all_derivs_n = cov_interp_all_derivs_n + cov_trunc_all_derivs_n

            cov_total_blocks_n = get_blocks_map(cov_total_all_derivs_n, (N_interp, N_interp))
            # for (ii, jj), cov_ij in cov_total_blocks_n.items():
            #     cov_total_blocks_n[ii, jj] += 1e-12 * np.eye(cov_ij.shape[0])
            std_total_vecs_n = get_std_map(cov_total_blocks_n)

            self._cov_total_all_derivs[n] = cov_total_all_derivs_n
            self._cov_total_blocks[n] = cov_total_blocks_n
            self._std_total_vecs[n] = std_total_vecs_n
