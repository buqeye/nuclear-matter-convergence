import numpy as np

hbar_c = 197.327  # MeV-fm
mass_proton = 938.272
mass_neutron = 939.565


def nuclear_density(momentum, degeneracy):
    R"""Computes the density of infinite matter in inverse fermi^3 given k_fermi

    Parameters
    ----------
    momentum : array
        The fermi momentum in inverse fermi
    degeneracy : int
        The degeneracy factor. Equals 2 for neutron matter and 4 for symmetric matter.
    """
    return degeneracy * momentum**3 / (6 * np.pi**2)


def fermi_momentum(density, degeneracy):
    R"""Computes the fermi momentum of infinite matter in inverse fermi

    Parameters
    ----------
    density : array
        The density in inverse fermi^3
    degeneracy : int
        The degeneracy factor. Equals 2 for neutron matter and 4 for symmetric matter.
    """
    return (6 * np.pi**2 * density / degeneracy)**(1./3)


def ratio_kf(momentum, breakdown=600):
    R"""
    Dimensionless expansion ratio of k_fermi to breakdown scale.

    Parameters
    ----------
    momentum : array
        The fermi momentum in inverse fermi
    breakdown: float, optional
        Breakdown scale (Lambda_b) in MeV; defaults to 600 MeV
    """
    return momentum.ravel() * hbar_c / breakdown


def ratio_density(density, breakdown=30):
    R"""THIS IS JUST FOR A TEST.
    Dimensionless expansion ratio of density to breakdown scale.

    Parameters
    ----------
    density : array
        The density in fm^{-3}
    breakdown: float, optional
        Breakdown scale (Lambda_b) in fm^{-3}; defaults to 30 fm^{-3}
    """
    return density.ravel() / breakdown


def Lb_prior(Lb, Lb_min=300, Lb_max=1000):
    R"""
    Uniform prior for the breakdown scale (Lambda_b aka Lb)
    """
    return np.where((Lb >= Lb_min) & (Lb <= Lb_max), 1 / Lb, 0.)


def Lb_logprior(Lb, Lb_min=300, Lb_max=1000):
    R"""
    Log uniform prior for the breakdown scale (Lambda_b aka Lb)
    """
    return np.where((Lb >= 300) & (Lb <= 1000), np.log(1 / Lb), -np.inf)


def figure_name(name, body, Lambda, fit_n2lo, fit_n3lo, breakdown, ref):
    R"""
    Generate filename for figures given specifications.

    Parameters
    ----------
    name :
    body :
    Lambda :
    fit_n2lo : index for fit at N2LO; only for NN+3N
    fit_n3lo : index for fit at N3LO; only for NN+3N
    breakdown :
    ref :
    """
    if body == 'NN+3N':
        full_name = name + f'_body-{body}_fits-{fit_n2lo}-{fit_n3lo}'
    else:
        full_name = name + f'_body-{body}_fits-0-0'
    full_name += f'_Lambda-{Lambda:.0f}_breakdown-{breakdown:.0f}_ref-{ref:.0f}.pdf'
    return full_name


def kf_derivative_wrt_density(kf, n):
    """Computes the derivative of kf with respect to density

    It is given by `kf / (3 * n)`

    Parameters
    ----------
    kf : array-like
        The fermi momentum in fm^-1
    n : array-like
        The density in fm^-3

    Returns
    -------
    d(kf)/d(n) : array-like
        In units of fm^2
    """
    return kf / (3 * n)


def kf_2nd_derivative_wrt_density(kf, n):
    """Computes the second derivative of kf with respect to density

    It is given by `-2 kf / (9 n**2)`

    Parameters
    ----------
    kf : array-like
        The fermi momentum in fm^-1
    n : array-like
        The density in fm^-3

    Returns
    -------
    d^2(kf)/d(n^2) : array-like
        In units of fm^5
    """
    return - 2 * kf / (9 * n**2)


def compute_pressure(n, kf, dE, wrt_kf=True):
    """Computes the pressure from the derivative of the energy per particle.

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    dE : array-like, shape = (N,)
        The derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `dE`. Defaults to `True`, so the derivative of `E` is with
        respect to `kf`.

    Returns
    -------
    P : array-like, shape = (N,)
        The pressure
    """
    P = n**2 * dE
    if wrt_kf:
        P *= kf_derivative_wrt_density(kf, n)
    return P


def compute_pressure_cov(n, kf, dE_cov, wrt_kf=True):
    """Computes the covariance of the pressure

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    dE_cov : array-like, shape = (N, N)
        The covariance of the derivative of the energy per particle.
        If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `dE`. Defaults to `True`, so the derivative of `E` is with
        respect to `kf`.

    Returns
    -------
    P_cov : array-like, shape = (N, N)
        The pressure covariance
    """
    n_mat = n[:, None] * n
    P_cov = n_mat**2 * dE_cov
    if wrt_kf:
        dk_dn = kf_derivative_wrt_density(kf, n)
        P_cov *= dk_dn[:, None] * dk_dn
    return P_cov


def compute_pressure_derivative_wrt_density(n, kf, dE, d2E, wrt_kf=True):
    """Computes the derivative of the pressure with respect to density.

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    dE : array-like, shape = (N,)
        The derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    d2E : array-like, shape = (N,)
        The 2nd derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `dE`. Defaults to `True`, so the derivative of `E` is with
        respect to `kf`.

    Returns
    -------
    dP/dn : array-like, shape = (N,)
        The derivative of the pressure with respect to density
    """
    if wrt_kf:
        dk_dn = kf_derivative_wrt_density(kf, n)
        d2k_dn2 = kf_2nd_derivative_wrt_density(kf, n)
        return (2 * n * dk_dn + n**2 * d2k_dn2) * dE + (n * dk_dn)**2 * d2E
    else:
        return 2 * n * dE + n**2 * d2E


def compute_slope(n, kf, dS2, wrt_kf=True):
    """Computes the slope from the derivative of the symmetry energy.

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    dS2 : array-like, shape = (N,)
        The derivative of symmetry energy. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `dS2`. Defaults to `True`, so the derivative of `S2` is with
        respect to `kf`.

    Returns
    -------
    L : array-like, shape = (N,)
        The slope parameter
    """
    L = 3 * n * dS2
    if wrt_kf:
        L *= kf_derivative_wrt_density(kf, n)
    return L


def compute_slope_cov(n, kf, dS2_cov, wrt_kf=True):
    """Computes the covariance of the slope

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    dS2_cov : array-like, shape = (N,)
        The covariance of the derivative of symmetry energy. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `dS2`. Defaults to `True`, so the derivative of `S2` is with
        respect to `kf`.

    Returns
    -------
    L_cov : array-like, shape = (N,)
        The covariance of the slope parameter
    """
    L_cov = 3**2 * n[:, None] * n * dS2_cov
    if wrt_kf:
        dk_dn = kf_derivative_wrt_density(kf, n)
        L_cov *= dk_dn[:, None] * dk_dn
    return L_cov


def compute_compressibility(n, kf, d2E, wrt_kf=True, dE=None):
    """Computes the compressibility from the derivatives of the energy per particle.

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    d2E : array-like, shape = (N,)
        The 2nd derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `d2E` and `dE`. Defaults to `True`,
        so the derivative of `E` is with respect to `kf`.
    dE : array-like, shape = (N,)
        The derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
        This is required if `wrt_kf is True`, but otherwise is unused.

    Returns
    -------
    K : array-like, shape = (N,)
        The compressibility
    """
    if wrt_kf:
        if dE is None:
            raise ValueError('dE must be given is wrt_kf is True')
        d2k_dn2 = kf_2nd_derivative_wrt_density(kf, n)
        dk_dn = kf_derivative_wrt_density(kf, n)
        d2E_dn2 = d2k_dn2 * dE + dk_dn**2 * d2E
    else:
        d2E_dn2 = d2E
    return 9 * n**2 * d2E_dn2


def compute_compressibility_cov(n, kf, d2E_cov, wrt_kf=True, dE_cov=None, dE_d2E_cov=None):
    """Computes the covariance of the compressibility

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    d2E_cov : array-like, shape = (N, N)
        The covariance of the 2nd derivative of the energy per particle.
        If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    wrt_kf : bool
        How to interpret the derivative `d2E` and `dE`. Defaults to `True`,
        so the derivative of `E` is with respect to `kf`.
    dE_cov : array-like, shape = (N, N)
        The covariance of the derivative of the energy per particle.
        This is required if `wrt_kf is True`, but otherwise is unused.
        If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    dE_d2E_cov : array-like, shape = (N, N)
        The covariance of the `dE` and `d2E`.
        This is required if `wrt_kf is True`, but otherwise is unused.
        If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.


    Returns
    -------
    K : array-like, shape = (N, N)
        The covariance of the compressibility
    """
    if wrt_kf:
        if dE_cov is None or dE_d2E_cov is None:
            raise ValueError('Both dE_cov and dE_d2E_cov are required if wrt_kf is True')
        d2k_dn2 = kf_2nd_derivative_wrt_density(kf, n)
        d2k_dn2_mat = d2k_dn2[:, None] * d2k_dn2
        dk_dn = kf_derivative_wrt_density(kf, n)
        dk_dn_mat = dk_dn[:, None] * dk_dn

        mixed_cov = d2k_dn2[:, None] * dk_dn**2 * dE_d2E_cov
        d2E_dn2_cov = d2k_dn2_mat * dE_cov + dk_dn_mat**2 * d2E_cov + mixed_cov + mixed_cov.T
    else:
        d2E_dn2_cov = d2E_cov
    n_mat = n[:, None] * n
    return 9**2 * n_mat**2 * d2E_dn2_cov


def compute_speed_of_sound(n, kf, E, dE, d2E, mass, wrt_kf=True):
    """Computes the speed of sound (squared) from the derivatives of the energy per particle

    Parameters
    ----------
    n : array-like, shape = (N,)
        The density in fm^-3
    kf : array-like, shape = (N,)
        The fermi momentum in fm^-1
    E : array-like, shape = (N,)
        The energy per particle.
    dE : array-like, shape = (N,)
        The derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    d2E : array-like, shape = (N,)
        The 2nd derivative of the energy per particle. If `wrt_kf is True`, then this derivative is
        assumed to be with respect to `kf`. Otherwise it is with respect to density.
    mass : float
        The mass of the particle
    wrt_kf : bool
        How to interpret the derivative `d2E` and `dE`. Defaults to `True`,
        so the derivative of `E` is with respect to `kf`.

    Returns
    -------
    c2 : array-like, shape = (N,)
        The speed of sound (squared)
    """
    dP_dn = compute_pressure_derivative_wrt_density(n, kf, dE, d2E, wrt_kf=wrt_kf)
    if wrt_kf:
        dk_dn = kf_derivative_wrt_density(kf, n)
        deps_dn = E + mass + n * dk_dn * dE
    else:
        deps_dn = E + mass + n * dE
    # return np.sqrt(dP_dn / deps_dn)
    return dP_dn / deps_dn
