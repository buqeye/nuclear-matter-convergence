import gsum as gm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import docrep
from .matter import nuclear_density, fermi_momentum
from os.path import join


docstrings = docrep.DocstringProcessor()

black = 'k'
softblack = 'k'
gray = '0.75'
darkgray = '0.5'
text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec=black, lw=0.8)


def compute_hyp_posterior(model, X, data, orders, max_idx, logprior, thetas, breakdowns):
    """Put this in the specific class?

    Parameters
    ----------
    model : gm.TruncationGP
    X :
    data
    orders
    max_idx
    logprior
    Lb

    Returns
    -------

    """
    model.fit(X, data[:, :max_idx+1], orders=orders[:max_idx+1])
    # trunc_kernel_theta = model.coeffs_process.kernel_.theta
    log_like = np.array([[model.log_marginal_likelihood(theta, breakdown=lb) for lb in breakdowns] for theta in thetas])
    log_like += logprior
    posterior = np.exp(log_like - np.max(log_like))
    posterior /= np.trapz(posterior, x=breakdowns)  # Normalize

    # bounds = np.zeros((2, 2))
    # for i, p in enumerate([0.68, 0.95]):
    #     bounds[i] = gm.hpd_pdf(pdf=posterior, alpha=p, x=Lb, disp=False)
    #
    # median = gm.median_pdf(pdf=posterior, x=Lb)
    return posterior


def compute_pdf_median_and_bounds(x, pdf, cred):
    bounds = np.zeros((len(cred), 2))
    for i, p in enumerate(cred):
        bounds[i] = gm.hpd_pdf(pdf=pdf, alpha=p, x=x, disp=False)
    median = gm.median_pdf(pdf=pdf, x=x)
    return median, bounds


def draw_summary_statistics(bounds68, bounds95, median, height=0, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(bounds68, [height, height], c=darkgray, lw=6, solid_capstyle='round')
    ax.plot(bounds95, [height, height], c=darkgray, lw=2, solid_capstyle='round')
    ax.plot([median], [height], c='white', marker='o', zorder=10, markersize=3)
    return ax


def offset_xlabel(ax):
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax


def joint_plot(ratio=1, height=3):
    """Taken from Seaborn JointGrid"""
    fig = plt.figure(figsize=(height, height))
    gsp = plt.GridSpec(ratio+1, ratio+1)

    ax_joint = fig.add_subplot(gsp[1:, :-1])
    ax_marg_x = fig.add_subplot(gsp[0, :-1], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gsp[1:, -1], sharey=ax_joint)

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    # Make the grid look nice
    from seaborn import utils
    # utils.despine(fig)
    utils.despine(ax=ax_marg_x, left=True)
    utils.despine(ax=ax_marg_y, bottom=True)
    fig.tight_layout(h_pad=0, w_pad=0)

    ax_marg_y.tick_params(axis='y', which='major', direction='out')
    ax_marg_x.tick_params(axis='x', which='major', direction='out')
    ax_marg_y.tick_params(axis='y', which='minor', direction='out')
    ax_marg_x.tick_params(axis='x', which='minor', direction='out')
    ax_marg_y.margins(x=0.1, y=0.)

    fig.subplots_adjust(hspace=0, wspace=0)

    return fig, ax_joint, ax_marg_x, ax_marg_y


def compute_2d_likelihood(model, X, data, orders, max_idx, logprior, Lb, ls):
    model.fit(X, data[:, :max_idx + 1], orders=orders[:max_idx + 1])
    log_like = np.array([
        [model.log_marginal_likelihood(theta=[np.log(ls_), ], breakdown=lb) for lb in Lb] for ls_ in ls
    ])
    log_like += logprior
    like = np.exp(log_like - np.max(log_like))
    # like /= np.trapz(like, x=Lb)  # Normalize

    ratio_like = np.trapz(like, x=ls, axis=0)
    ls_like = np.trapz(like, x=Lb, axis=-1)

    # Normalize them
    ratio_like /= np.trapz(ratio_like, x=Lb, axis=0)
    ls_like /= np.trapz(ls_like, x=ls, axis=0)
    return like, ratio_like, ls_like


def plot_2d_joint(ls_vals, Lb_vals, like_2d, like_ls, like_Lb, data_str=r'\vec{\mathbf{y}}_k)'):

    with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
        cmap_name = 'Blues'
        cmap = mpl.cm.get_cmap(cmap_name)

        # Setup axes
        fig, ax_joint, ax_marg_x, ax_marg_y = joint_plot(ratio=5, height=3.4)

        # Plot contour
        ax_joint.contour(ls_vals, Lb_vals, like_2d.T,
                         levels=[np.exp(-0.5*r**2) for r in np.arange(9, 0, -0.5)] + [0.999],
                         cmap=cmap_name, vmin=-0.05, vmax=0.8, zorder=1)

        # Now plot the marginal distributions
        ax_marg_y.plot(like_Lb, Lb_vals, c=cmap(0.8), lw=1)
        ax_marg_y.fill_betweenx(Lb_vals, np.zeros_like(like_Lb),
                                like_Lb, facecolor=cmap(0.2), lw=1)
        ax_marg_x.plot(ls_vals, like_ls, c=cmap(0.8), lw=1)
        ax_marg_x.fill_between(ls_vals, np.zeros_like(ls_vals),
                               like_ls, facecolor=cmap(0.2), lw=1)

        # Formatting
        ax_joint.set_xlabel(r'$\ell$')
        ax_joint.set_ylabel(r'$\Lambda_b$')
        ax_joint.margins(x=0, y=0.)
        ax_marg_x.set_ylim(bottom=0);
        ax_marg_y.set_xlim(left=0);
        ax_joint.text(0.95, 0.95, rf'pr$(\ell, \Lambda_b \,|\, {data_str}$)', ha='right', va='top',
                      transform=ax_joint.transAxes,
                      bbox=text_bbox
                     )
        ax_joint.tick_params(direction='in')

        plt.show()
        return fig


@docstrings.get_sectionsf('ConvergenceAnalysis')
@docstrings.dedent
class ConvergenceAnalysis:
    R"""

    Parameters
    ----------
    X : ndarray
    y : ndarray
    orders : ndarray
    train : ndarray
    valid : ndarray
    ref : float or callable
    ratio : float or callable
    ignore_orders : ndarray, optional
    colors : ndarray, optional
    kwargs : dict
    """

    def __init__(self, X, y, orders, train, valid, ref, ratio, *, ignore_orders=None, colors=None, **kwargs):
        self.X = X
        self.y = y
        self.orders = orders

        self.train = train
        self.valid = valid
        self.X_train = X[train]
        self.X_valid = X[valid]
        self.y_train = y[train]
        self.y_valid = y[valid]

        self.ratio = ratio
        self.ref = ref
        self.ignore_orders = ignore_orders
        if ignore_orders is None:
            ignore_mask = np.ones_like(orders, dtype=bool)
        else:
            ignore_mask = ~np.isin(orders, ignore_orders)
        self.ignore_mask = ignore_mask

        self.colors = colors
        self.kwargs = kwargs

    def compute_coefficients(self, **kwargs):
        ratio = self.ratio(**kwargs)
        c = gm.coefficients(self.y, ratio, self.ref, self.orders)[:, self.ignore_mask]
        return c

    def plot_coefficients(self, *args, **kwargs):
        raise NotImplementedError

    def plot_pchol(self):
        pass

    def plot_md_squared(self):
        pass


@docstrings.dedent
class MatterConvergenceAnalysis(ConvergenceAnalysis):
    """A convenience class to compute quantities related to nuclear matter convergence

    Parameters
    ----------
    %(ConvergenceAnalysis.parameters)s
    system : str
    """

    system_strings = dict(
        neutron='neutron',
        symmetric='symmetric',
        difference='difference',
    )
    system_math_strings = dict(
        neutron='E/N',
        symmetric='E/A',
        difference='S_n',
    )

    def __init__(self, X, y, orders, train, valid, ref, ratio, *, system='neutron',
                 fit_n2lo=None, fit_n3lo=None, Lambda=None, body=None, savefig=False, **kwargs):
        super(MatterConvergenceAnalysis, self).__init__(
            X, y, orders, train, valid, ref, ratio, **kwargs)
        self.system = system
        self.fit_n2lo = fit_n2lo
        self.fit_n3lo = fit_n3lo
        self.Lambda = Lambda
        self.body = body
        self.savefig = savefig
        self.fig_path = 'figures'

    def compute_density(self, kf):
        degeneracy = None
        if self.system == 'neutron':
            degeneracy = 2
        elif self.system == 'symmetric':
            degeneracy = 4
        elif self.system == 'difference':
            raise ValueError('not sure what to do for symmetry energy')

        return nuclear_density(kf, degeneracy)

    def compute_momentum(self, density):
        degeneracy = None
        if self.system == 'neutron':
            degeneracy = 2
        elif self.system == 'symmetric':
            degeneracy = 4
        elif self.system == 'difference':
            raise ValueError('not sure what to do for symmetry energy')

        return fermi_momentum(density, degeneracy)

    def compute_underlying_graphical_diagnostic(self):
        coeffs = self.compute_coefficients()
        process = gm.ConjugateGaussianProcess(**self.kwargs)
        mean = process.mean(self.X_valid)
        cov = process.cov(self.X_valid)
        graph = gm.GraphicalDiagnostic(coeffs[self.valid], mean, cov, colors=self.colors, gray=gray, black=softblack)
        return graph

    def figure_name(self, prefix, breakdown=None):
        body = self.body
        fit_n2lo = self.fit_n2lo
        fit_n3lo = self.fit_n3lo
        Lambda = self.Lambda
        ref = self.ref
        system = self.system_strings[self.system]

        full_name = prefix + f'sys-{system}_body-{body}'
        if body == 'NN+3N':
            full_name += f'_fits-{fit_n2lo}-{fit_n3lo}'
        else:
            full_name += f'_fits-0-0'
        full_name += f'_Lambda-{Lambda:.0f}'
        if breakdown is not None:
            full_name += f'_breakdown-{breakdown:.0f}'
        else:
            full_name += f'_breakdown-None'
        full_name += f'_ref-{ref:.0f}'
        full_name = join(self.fig_path, full_name)
        return full_name

    def plot_coefficients(self, breakdown, ax=None):
        coeffs = self.compute_coefficients(breakdown=breakdown)
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
        kf = self.X.ravel()
        d = self.compute_density(kf)
        ax2 = ax.twiny()
        train = self.train
        colors = self.colors
        for i, n in enumerate(self.orders):
            ax.plot(kf, coeffs[:, i], c=colors[i], label=fr'$c_{{{n}}}$')
            ax.plot(kf[train], coeffs[train, i], marker='o', ls='', c=colors[i])

        ax.legend()
        ax2.plot(d, np.zeros_like(d), ls='--', c=gray, zorder=0)  # Dummy data to set up ticks
        ax2.set_xlabel(r'Density $n$ (fm$^{-3}$)')
        # ax.set_ylabel(r'Energy per Neutron $E/N$')
        ax.set_xlabel(r'Fermi Momentum $k_\mathrm{F}$ (fm$^{-1}$)')

        if self.savefig:
            fig = plt.gcf()
            fig.savefig(self.figure_name('coeffs', breakdown=breakdown))
        return ax
