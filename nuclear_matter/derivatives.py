import gptools
import numpy as np
from sympy import symbols, diff, lambdify
from findiff import FinDiff


class CustomKernel(gptools.Kernel):
    """A Custom GPTools kernel that wraps an arbitrary function f with a compatible signature

    Parameters
    ----------
    f : callable
        A positive semidefinite kernel function that takes f(Xi, Xj, ni, nj) where ni and nj are
        integers for the number of derivatives to take with respect to Xi or Xj. It should return
        an array of Xi.shape[0]
    *args
        Args passed to the Kernel class
    **kwargs
        Kwargs passed to the Kernel class
    """

    def __init__(self, f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

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

        value = np.NaN * np.ones(Xi.shape[0])
        for (i, j), mask in dmasks.items():
            value[mask] = self.f(Xi[mask], Xj[mask], i, j)
        return value


#         print(ni, nj, flush=True)
# Only works for 1d X.
#         return np.array([
#             self.f(x_i, x_j, int(n_i[0]), int(n_j[0]))
#             for x_i, x_j, n_i, n_j in zip(Xi, Xj, ni, nj)
#         ])


def kernel_scale_sympy(lowest_order=4, highest_order=None, include_3bf=False):
    """Creates a sympy object that is the convergence part of the GP kernel

    Parameters
    ----------
    lowest_order
    highest_order
    include_3bf
    """
    k_f1, k_f2, y_ref, Lambda_b, = symbols('k_f1 k_f2 y_ref Lambda_b')
    hbar_c = 197.3269718  # MeV fm
    Q1 = hbar_c * k_f1 / Lambda_b
    Q2 = hbar_c * k_f2 / Lambda_b
    num = (Q1 * Q2) ** lowest_order
    if highest_order is not None:
        num = num - (Q1 * Q2) ** (highest_order + 1)
    kernel_scale = y_ref ** 2 * num / (1 - Q1 * Q2)

    if include_3bf and (highest_order is None or highest_order >= 3):
        lowest_order_3bf = lowest_order if lowest_order >= 3 else 3
        num_3bf = (Q1 * Q2) ** lowest_order_3bf
        if highest_order is not None:
            num_3bf = num_3bf - (Q1 * Q2) ** (highest_order + 1)

        # kf_3bf_order = 3
        kf_3bf_order = 1  # Actually, linear doesn't look so bad
        kernel_scale += (k_f1 * k_f2)**kf_3bf_order * y_ref ** 2 * num_3bf / (1 - Q1 * Q2)

    return k_f1, k_f2, Lambda_b, y_ref, kernel_scale


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

    def __init__(self, breakdown=600, ref=16, lowest_order=4, highest_order=None, include_3bf=False):

        # TODO: Fix the reference scale for 3bf
        self.breakdown = breakdown
        self.ref = ref
        self.lowest_order = lowest_order
        self.highest_order = highest_order

        k_f1, k_f2, Lambda_b, y_ref, kernel_scale = kernel_scale_sympy(
            lowest_order=lowest_order, highest_order=highest_order, include_3bf=include_3bf
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
        try:
            K = K.astype('float')
        except:
            pass
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


class ObservableContainer:

    def __init__(
            self, density, kf, y, orders, density_interp, kf_interp,
            std, ls, breakdown, err_y=0, derivs=(0, 1, 2), include_3bf=True
    ):

        self.density = density
        self.kf = kf
        self.Kf = Kf = kf[:, None]

        self.density_interp = density_interp
        self.kf_interp = kf_interp
        self.Kf_interp = Kf_interp = kf_interp[:, None]

        self.y = y
        self.N_interp = N_interp = len(kf_interp)
        self.err_y = err_y
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

        self.coeff_kernel = gptools.SquaredExponentialKernel(
            initial_params=[std, ls], fixed_params=[True, True])
        for i, n in enumerate(orders):
            first_omitted = n + 1
            if first_omitted == 1:
                first_omitted += 1  # the Q^1 contribution is zero, so bump to Q^2
            _kern_lower = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, lowest_order=0, highest_order=n, include_3bf=include_3bf
            ))
            kern_interp = _kern_lower * self.coeff_kernel
            _kern_upper = CustomKernel(ConvergenceKernel(
                breakdown=breakdown, lowest_order=first_omitted, include_3bf=include_3bf
            ))
            kern_trunc = _kern_upper * self.coeff_kernel

            try:
                err_y_i = err_y[i]
            except TypeError:
                err_y_i = err_y

            # Interpolating processes
            gp_interp = gptools.GaussianProcess(kern_interp)
            gp_interp.add_data(Kf, y[:, i], err_y=err_y_i)
            self.gps_interp[n] = gp_interp

            # Finite difference:
            self._dy_dn[n] = d_dn(y[:, i])
            self._d2y_dn2[n] = d2_dn2(y[:, i])
            self._dy_dk[n] = d_dk(y[:, i])
            self._d2y_dk2[n] = d2_dk2(y[:, i])
            self._y_dict[n] = y[:, i]

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

    def get_pred(self, order, deriv):
        return self._y_interp_vecs[order][deriv]

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
        samples = gp.draw_sample(Xstar=self.Kf_interp, num_samp=num_samp, mean=mean, cov=cov)
        # change it to: n_derivs, N_interp, num_samp
        sample_blocks = extract_blocks(samples, blocksize=(self.N_interp, samples.shape[-1]))
        sample_dict = {}
        # self.derivs = [0, 1, 2]
        # Put into dict for access via derivative value
        for i, d in enumerate(self.derivs):
            sample_dict[d] = sample_blocks[i]
        return sample_dict

    def predict(self, Kf, order, derivs=None, include_trunc=True):
        if derivs is None:
            derivs = self.derivs
        y_interp, cov = predict_with_derivatives(
            self.gps_interp[order], X=Kf, n=derivs, return_cov=True
        )
        if include_trunc:
            cov += predict_with_derivatives(
                self.gps_trunc[order], X=Kf, n=derivs, only_cov=True
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
