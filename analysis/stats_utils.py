import gsum as gm
import numpy as np
from numpy import ndarray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import docrep
import seaborn as sns
import pandas as pd
from matter import nuclear_density, fermi_momentum
from os.path import join


docstrings = docrep.DocstringProcessor()

black = 'k'
softblack = 'k'
gray = '0.75'
darkgray = '0.5'
text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec=black, lw=0.8)


def compute_breakdown_posterior(model, X, data, orders, max_idx, logprior, breakdowns, lengths=None):
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
    pdf : ndarray, shape = (N,)
    """
    model.fit(X, data[:, :max_idx+1], orders=orders[:max_idx+1])
    if lengths is None:
        log_ell = model.coeffs_process.kernel_.theta
        lengths = np.exp(log_ell)
    else:
        log_ell = np.log(lengths)
    log_like = np.array([[model.log_marginal_likelihood([t], breakdown=lb) for lb in breakdowns] for t in log_ell])
    log_like += logprior
    posterior_2d = np.exp(log_like - np.max(log_like))

    breakdown_pdf = np.trapz(posterior_2d, x=lengths, axis=0)
    breakdown_pdf /= np.trapz(breakdown_pdf, x=breakdowns)  # Normalize
    return breakdown_pdf


def compute_pdf_median_and_bounds(x, pdf, cred):
    bounds = np.zeros((len(cred), 2))
    for i, p in enumerate(cred):
        bounds[i] = gm.hpd_pdf(pdf=pdf, alpha=p, x=x, disp=False)
    median = gm.median_pdf(pdf=pdf, x=x)
    return median, bounds


def draw_summary_statistics(bounds68, bounds95, median, height=0., ax=None):
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


def compute_2d_posterior(model, X, data, orders, max_idx, breakdown, ls, logprior=None):
    model.fit(X, data[:, :max_idx + 1], orders=orders[:max_idx + 1])
    log_like = np.array([
        [model.log_marginal_likelihood(theta=[np.log(ls_), ], breakdown=lb) for lb in breakdown] for ls_ in ls
    ])
    if logprior is not None:
        log_like += logprior
    joint_pdf = np.exp(log_like - np.max(log_like))
    # like /= np.trapz(like, x=Lb)  # Normalize

    ratio_pdf = np.trapz(joint_pdf, x=ls, axis=0)
    ls_pdf = np.trapz(joint_pdf, x=breakdown, axis=-1)

    # Normalize them
    ratio_pdf /= np.trapz(ratio_pdf, x=breakdown, axis=0)
    ls_pdf /= np.trapz(ls_pdf, x=ls, axis=0)
    return joint_pdf, ratio_pdf, ls_pdf


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


def pdfplot(
        x, y, pdf, data, hue=None, order=None, hue_order=None, cut=1e-2, linewidth=None,
        palette=None, saturation=1., ax=None, margin=None,
):
    R"""Like seaborn's violinplot, but takes PDF values rather than tabular data.

    Parameters
    ----------
    x :
    y :
    pdf :
    data :
    hue :
    order :
    hue_order :
    cut :
    linewidth :
    palette :
    saturation :
    ax : mpl.axes.Axis
    margin :
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.4))

    y_vals = data[y].unique()
    if order is not None:
        y_vals = order

    legend_vals = y_vals
    hue_vals = [None]
    n_colors = len(y_vals)
    if hue is not None:
        hue_vals = data[hue].unique()
        if hue_order is not None:
            hue_vals = hue_order
        legend_vals = hue_vals
        n_colors = len(hue_vals)

    if isinstance(palette, str) or palette is None:
        colors = sns.color_palette(palette, n_colors=n_colors, desat=saturation)
    elif isinstance(palette, list):
        colors = palette
    else:
        raise ValueError('palette must be str or list')

    if margin is None:
        _, margin = plt.margins()

    offset = 1.
    minor_ticks = []
    major_ticks = []
    for i, y_val in enumerate(y_vals):
        max_height_hue = offset - margin
        for j, hue_val in enumerate(hue_vals):
            mask = data[y] == y_val
            if hue is not None:
                mask = mask & (data[hue] == hue_val)
                color = colors[j]
            else:
                color = colors[i]
            df = data[mask]
            # print(df)
            # print(y_val, hue_val)
            x_vals = df[x].values
            pdf_vals = df[pdf].values
            # Assumes normalized
            median, bounds = compute_pdf_median_and_bounds(x_vals, pdf_vals, cred=[0.68, 0.95])
            pdf_vals = pdf_vals / (1. * np.max(pdf_vals))  # Scale so they're all the same height
            # Make the lines taper off
            x_vals = x_vals[pdf_vals > cut]
            pdf_vals = pdf_vals[pdf_vals > cut]
            offset -= (1 + margin)
            # if ((i != 0) | (j != 0)):
            #     offset -= margin
            # Plot and fill posterior, and add summary statistics
            ax.plot(x_vals, pdf_vals + offset, c=darkgray, lw=linewidth)
            ax.fill_between(x_vals, offset, pdf_vals + offset, facecolor=color)
            draw_summary_statistics(*bounds, median, ax=ax, height=offset)
        min_height_hue = offset
        minor_ticks.append(offset - margin/2.)
        major_ticks.append((max_height_hue + min_height_hue) / 2.)

    minor_ticks = minor_ticks[:-1]
    # Plot formatting
    ax.set_yticks(major_ticks, minor=False)
    ax.set_yticks(minor_ticks, minor=True)
    ax.set_yticklabels(y_vals, fontdict=dict(verticalalignment='center'))
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(which='major', length=0)
    ax.tick_params(which='minor', length=7, right=True)
    # ax.set_xlim(0, 1200)
    # ax.set_xticks([0, 300, 600, 900, 1200])
    ax.set_xlabel(x)
    # ax.grid(axis='x')
    ax.set_axisbelow(True)

    if hue is not None:
        legend_elements = [
            Patch(facecolor=color, edgecolor=darkgray, label=leg_val) for color, leg_val in zip(colors, legend_vals)
        ]
        ax.legend(handles=legend_elements, loc='best')
    return ax


def matter_pdf_dfs(analyses, max_idx, breakdown, ls, logprior=None):
    dfs_breakdown = []
    dfs_ls = []
    dfs_joint = []
    for analysis in analyses:
        for idx in np.atleast_1d(max_idx):
            joint_pdf, breakdown_pdf, ls_pdf = analysis.compute_breakdown_ls_posterior(
                breakdown, ls, max_idx=idx, logprior=logprior)

            df_breakdown = pd.DataFrame(np.array([breakdown, breakdown_pdf]).T, columns=[r'$\Lambda_b$ (MeV)', 'pdf'])
            df_breakdown['Order'] = fr'N$^{idx}$LO'
            df_breakdown['system'] = fr'${analysis.system_math_string}$'
            dfs_breakdown.append(df_breakdown)

            df_ls = pd.DataFrame(np.array([ls, ls_pdf]).T, columns=[r'$\ell$', 'pdf'])
            df_ls['Order'] = fr'N$^{idx}$LO'
            df_ls['system'] = fr'${analysis.system_math_string}$'
            dfs_ls.append(df_ls)

            X = gm.cartesian(ls, breakdown)
            df_joint = pd.DataFrame(X, columns=[r'$\ell$', r'$\Lambda_b$ (MeV)'])
            df_joint['pdf'] = joint_pdf.ravel()
            df_joint['Order'] = fr'N$^{idx}$LO'
            df_joint['system'] = fr'${analysis.system_math_string}$'
            dfs_joint.append(df_joint)
    df_breakdown = pd.concat(dfs_breakdown, ignore_index=True)
    df_ls = pd.concat(dfs_ls, ignore_index=True)
    df_joint = pd.concat(dfs_joint, ignore_index=True)
    return df_joint, df_breakdown, df_ls


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

    def __init__(self, X, y, orders, train, valid, ref, ratio, *, excluded=None, colors=None, **kwargs):
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
        self.excluded = excluded
        if excluded is None:
            ignore_mask = np.ones_like(orders, dtype=bool)
        else:
            ignore_mask = ~np.isin(orders, excluded)
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
    fit_n2lo : str
    fit_n3lo : str
    Lambda : int
    body : str
    savefig : bool
    """

    system_strings = dict(
        neutron='neutron',
        symmetric='symmetric',
        difference='difference',
    )
    system_math_strings = dict(
        neutron='E/N',
        symmetric='E/A',
        difference='S_2',
    )

    def __init__(self, X, y, orders, train, valid, ref, ratio, *, system='neutron',
                 fit_n2lo=None, fit_n3lo=None, Lambda=None, body=None, savefig=False, **kwargs):
        super().__init__(
            X, y, orders, train, valid, ref, ratio, **kwargs)
        self.system = system
        self.fit_n2lo = fit_n2lo
        self.fit_n3lo = fit_n3lo
        self.Lambda = Lambda
        self.body = body
        self.savefig = savefig
        self.fig_path = 'figures'
        self.system_math_string = self.system_math_strings[system]

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

    def compute_breakdown_ls_posterior(self, breakdown, ls, max_idx=None, logprior=None):
        orders = self.orders[:max_idx + 1]
        model = gm.TruncationGP(ref=self.ref, ratio=self.ratio, excluded=self.excluded, **self.kwargs)
        X = self.X_train
        data = self.y_train
        joint_pdf, Lb_pdf, ls_pdf = compute_2d_posterior(
            model, X, data, orders, max_idx, breakdown, ls, logprior=logprior
        )
        return joint_pdf, Lb_pdf, ls_pdf

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
        if self.system == 'neutron':
            y_label = fr'Energy per Neutron '
        elif self.system == 'symmetric':
            y_label = 'Energy per Particle '
        elif self.system == 'difference':
            y_label = 'Symmetry Energy '
        else:
            raise ValueError('system has wrong value')
        y_label += fr'${self.system_math_strings[self.system]}$'
        ax.set_ylabel(y_label)
        ax.set_xlabel(r'Fermi Momentum $k_\mathrm{F}$ (fm$^{-1}$)')

        if self.savefig:
            fig = plt.gcf()
            fig.savefig(self.figure_name('coeffs', breakdown=breakdown))
        return ax
