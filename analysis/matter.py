import numpy as np

hbar_c = 197.327  # MeV-fm


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
