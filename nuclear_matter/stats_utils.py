import gsum as gm
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend import Legend
import docrep
import seaborn as sns
from seaborn import utils
import pandas as pd
from .matter import nuclear_density, fermi_momentum, ratio_kf
from .graphs import confidence_ellipse, confidence_ellipse_mean_cov
from os.path import join
from scipy import stats
from copy import deepcopy
from os import path

docstrings = docrep.DocstringProcessor()
docstrings.get_sections(str(gm.ConjugateGaussianProcess.__doc__), 'ConjugateGaussianProcess')

black = 'k'
softblack = 'k'
gray = '0.75'
darkgray = '0.5'
text_bbox = dict(boxstyle='round', fc=(1, 1, 1, 0.6), ec=black, lw=0.8)


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


Legend.update_default_handler_map({Ellipse: HandlerEllipse()})


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
    breakdowns
    lengths

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
    R"""Computes the median and credible intervals for a 1d pdf

    Parameters
    ----------
    x : 1d array
        The input variable
    pdf : 1d array
        The normalized pdf
    cred : Iterable
        The credible intervals in the range (0, 1)

    Returns
    -------
    median : float
    bounds : ndarray, shape = (len(cred), 2)
    """
    bounds = np.zeros((len(cred), 2))
    for i, p in enumerate(cred):
        bounds[i] = gm.hpd_pdf(pdf=pdf, alpha=p, x=x)
    median = gm.median_pdf(pdf=pdf, x=x)
    return median, bounds


def draw_summary_statistics(bounds68, bounds95, median, height=0., linewidth=1., ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(bounds68, [height, height], c=darkgray, lw=3*linewidth, solid_capstyle='round')
    ax.plot(bounds95, [height, height], c=darkgray, lw=linewidth, solid_capstyle='round')
    ax.plot([median], [height], c='white', marker='o', zorder=10, markersize=1.5*linewidth)
    return ax


def offset_xlabel(ax):
    ax.set_xticks([0])
    ax.set_xticklabels(labels=[0], fontdict=dict(color='w'))
    ax.tick_params(axis='x', length=0)
    return ax


def joint_plot(ratio=1, height=3.):
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


def compute_2d_posterior(model, X, data, orders, breakdown, ls=None, logprior=None, max_idx=None):
    R"""

    Parameters
    ----------
    model : gm.TruncationGP
    X : ndarray, shape = (N,None)
    data : ndarray, shape = (N,[n_curves])
    orders : ndarray, shape = (n_curves,)
    max_idx : ndarray, shape = (n_orders,)
    breakdown : ndarray, shape = (n_breakdown,)
    ls : ndarray, shape = (n_ls,)
    logprior : ndarray, optional, shape = (n_ls, n_breakdown)

    Returns
    -------
    joint_pdf : ndarray
    ratio_pdf : ndarray
    ls_pdf : ndarray
    """
    if max_idx is not None:
        data = data[:, :max_idx + 1]
        orders = orders[:max_idx + 1]
    model.fit(X, data, orders=orders)
    if ls is None:
        ls = np.exp(model.coeffs_process.kernel_.theta)
        print('Setting ls to', ls)
    ls = np.atleast_1d(ls)
    # log_like = np.array([
    #     [model.log_marginal_likelihood(theta=[np.log(ls_), ], breakdown=lb) for lb in breakdown] for ls_ in ls
    # ])
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    log_like = np.array(
        Parallel(n_jobs=num_cores, prefer='processes')(
            delayed(model.log_marginal_likelihood)(theta=[np.log(ls_), ], breakdown=lb)
            for ls_ in ls for lb in breakdown
        )
    ).reshape(len(ls), len(breakdown))
    if logprior is not None:
        log_like += logprior
    joint_pdf = np.exp(log_like - np.max(log_like))

    if len(ls) > 1:
        ratio_pdf = np.trapz(joint_pdf, x=ls, axis=0)
    else:
        ratio_pdf = np.squeeze(joint_pdf)
    ls_pdf = np.trapz(joint_pdf, x=breakdown, axis=-1)

    # Normalize them
    ratio_pdf /= np.trapz(ratio_pdf, x=breakdown, axis=0)
    if len(ls) > 1:
        ls_pdf /= np.trapz(ls_pdf, x=ls, axis=0)
    return joint_pdf, ratio_pdf, ls_pdf


def plot_2d_joint(ls_vals, Lb_vals, like_2d, like_ls, like_Lb, data_str=r'\vec{\mathbf{y}}_k)',
                  xlabel=None, ylabel=None):
    if data_str is None:
        data_str = r'\vec{\mathbf{y}}_k)'
    from matplotlib.cm import get_cmap
    with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
        cmap_name = 'Blues'
        cmap = get_cmap(cmap_name)

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
        ax_joint.set_xlabel(xlabel)
        ax_joint.set_ylabel(ylabel)
        ax_joint.margins(x=0, y=0.)
        ax_marg_x.set_ylim(bottom=0)
        ax_marg_y.set_xlim(left=0)
        ax_joint.text(
            0.95, 0.95, rf'pr$(\ell, \Lambda_b \,|\, {data_str}$)', ha='right', va='top',
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
    x : str
        The column of the DataFrame to use as the x axis. The pdfs are a function of this variable.
    y : str
        The column of the DataFrame to use as the y axis. A pdf will be drawn for each unique value in data[y].
    pdf : str
        The column of the DataFrame to use as the pdf values.
    data : pd.DataFrame
        The DataFrame containing the pdf data
    hue : str, optional
        Splits data[y] up by the value of hue, and plots each pdf separately as a specific color.
    order : list, optional
        The order in which to plot the y values, from top to bottom
    hue_order : list, optional
        The order in which to plot the hue values, from top to bottom.
    cut : float, optional
        The value below which the pdfs will not be shown. This is taken as a fraction of the total height of each pdf.
    linewidth : float, optional
        The linewidth of the pdf lines
    palette : str, list, optional
        The color palette to fill underneath the curves
    saturation : float, optional
        The level of saturation for the color palette. Only works if the palette is a string recognized by
        sns.color_palette
    ax : matplotlib.axes.Axes
        The axis on which to draw the plot
    margin : float, optional
        The vertical margin between each pdf.
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
            x_vals = df[x].values
            pdf_vals = df[pdf].values.copy()
            pdf_vals /= np.trapz(pdf_vals, x_vals)
            # Assumes normalized
            median, bounds = compute_pdf_median_and_bounds(
                x=x_vals, pdf=pdf_vals, cred=[0.68, 0.95]
            )
            pdf_vals /= (1. * np.max(pdf_vals))  # Scale so they're all the same height
            # Make the lines taper off
            x_vals = x_vals[pdf_vals > cut]
            pdf_vals = pdf_vals[pdf_vals > cut]
            offset -= (1 + margin)

            # Plot and fill posterior, and add summary statistics
            ax.plot(x_vals, pdf_vals + offset, c=darkgray, lw=linewidth)
            ax.fill_between(x_vals, offset, pdf_vals + offset, facecolor=color)
            draw_summary_statistics(*bounds, median, ax=ax, height=offset, linewidth=1.5*linewidth)
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
    ax.set_xlabel(x)
    ax.set_axisbelow(True)

    if hue is not None:
        legend_elements = [
            Patch(facecolor=color, edgecolor=darkgray, label=leg_val) for color, leg_val in zip(colors, legend_vals)
        ]
        ax.legend(handles=legend_elements, loc='best')
    return ax


def joint2dplot(ls_df, breakdown_df, joint_df, system, order, data_str=None):
    ls_df = ls_df[(ls_df['system'] == system) & (ls_df['Order'] == order)]
    breakdown_df = breakdown_df[(breakdown_df['system'] == system) & (breakdown_df['Order'] == order)]
    joint_df = joint_df[(joint_df['system'] == system) & (joint_df['Order'] == order)]
    ls = ls_df[r'$\ell$ (fm$^{-1}$)']
    breakdown = breakdown_df[r'$\Lambda_b$ (MeV)']
    joint = joint_df['pdf'].values.reshape(len(ls), len(breakdown))
    fig = plot_2d_joint(
        ls_vals=ls, Lb_vals=breakdown, like_2d=joint,
        like_ls=ls_df['pdf'].values, like_Lb=breakdown_df['pdf'].values,
        data_str=data_str, xlabel=r'$\ell$ (fm$^{-1}$)', ylabel=r'$\Lambda_b$ (MeV)',
    )
    return fig


def minimum_samples(mean, cov, n=5000, x=None):
    gp = stats.multivariate_normal(mean=mean, cov=cov)
    samples = gp.rvs(n)
    min_idxs = np.argmin(samples, axis=1)
    min_y = np.min(samples, axis=1)
    if x is not None:
        min_x = x[min_idxs]
        return min_x, min_y
    return min_idxs, min_y


# def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
#     """
#     Create a plot of the covariance confidence ellipse of *x* and *y*.
#
#     Parameters
#     ----------
#     x, y : array-like, shape (n, )
#         Input data.
#     ax : matplotlib.axes.Axes
#         The axes object to draw the ellipse into.
#     n_std : float
#         The number of standard deviations to determine the ellipse's radii.
#     facecolor : str
#         The color of the ellipse
#
#     Returns
#     -------
#     matplotlib.patches.Ellipse
#
#     Other parameters
#     ----------------
#     kwargs : `~matplotlib.patches.Patch` properties
#     """
#     import matplotlib.transforms as transforms
#     if x.size != y.size:
#         raise ValueError("x and y must be the same size")
#
#     cov = np.cov(x, y)
#     pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#     # Using a special case to obtain the eigenvalues of this
#     # two-dimensional dataset.
#     ell_radius_x = np.sqrt(1 + pearson)
#     ell_radius_y = np.sqrt(1 - pearson)
#     ellipse = Ellipse(
#         (0, 0),
#         width=ell_radius_x * 2,
#         height=ell_radius_y * 2,
#         facecolor=facecolor,
#         **kwargs
#     )
#
#     # Calculating the standard deviation of x from
#     # the square root of the variance and multiplying
#     # with the given number of standard deviations.
#     scale_x = np.sqrt(cov[0, 0]) * n_std
#     mean_x = np.mean(x)
#
#     # calculating the standard deviation of y ...
#     scale_y = np.sqrt(cov[1, 1]) * n_std
#     mean_y = np.mean(y)
#
#     trans = transforms.Affine2D() \
#         .rotate_deg(45) \
#         .scale(scale_x, scale_y) \
#         .translate(mean_x, mean_y)
#
#     ellipse.set_transform(trans + ax.transData)
#     # sns.kdeplot(x, y, ax=ax)
#     scat_color = darken_color(facecolor, 0.5)
#     ax.plot(x, y, ls='', marker='.', markersize=0.6, color=scat_color)
#     ax.add_patch(ellipse)
#     return ellipse


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def darken_color(color, amount=0.5):
    """
    Darken the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    return lighten_color(color, 1./amount)


@docstrings.get_sectionsf('ConvergenceAnalysis')
@docstrings.dedent
class ConvergenceAnalysis:
    R"""A generic class for studying convergence of observables.

    This is meant to provide the framework for particular analyses, which should subclass this class.

    Parameters
    ----------
    X : ndarray, shape = (N,p)
        The feature matrix
    y : ndarray, shape = (N, n_curves)
        The response curves
    orders : ndarray, shape = (n_orders,)
    train : ndarray, shape = (N,)
        A boolean array that is `True` if that point is to be used to train the convergence model.
    valid : ndarray, shape = (N,)
        A boolean array that is `True` if that point is to be used to validate the convergence model.
    ref : float or callable
        The reference scale
    ratio : float or callable
        The ratio Q
    excluded : ndarray, optional
        The orders for which the coefficients should not be used in training the convergence model.
    colors : ndarray, optional
        Colors for plotting orders and their diagnostics.

    Other Parameters
    ----------------
    %(ConjugateGaussianProcess.parameters)s
    """

    def __init__(self, X, y2, y3, orders, train, valid, ref2, ref3, ratio, body, *, excluded=None, colors=None, **kwargs):
        self.X = X
        self.orders_original = np.atleast_1d(orders)

        marker_list = ['^', 'X', 'o', 's']
        markerfillstyle_2bf = 'full'
        markerfillstyle_3bf = 'left'
        colors_original = colors

        if body == 'Appended':
            print('Appending 2bf and 3bf predictions...')
            try:
                ref3_vals = ref3(X)
            except TypeError:
                ref3_vals = ref3
            try:
                ratio_vals = ratio(X)
            except TypeError:
                ratio_vals = ratio
            c2 = gm.coefficients(y2, ratio_vals, ref2, orders)
            c3 = gm.coefficients(y3-y2, ratio_vals, ref3_vals, orders)
            c = []
            colors_all = []
            orders_all = []
            markers = []
            markerfillstyles = []
            n_bodies = []

            for i, n in enumerate(orders):
                c.append(c2[:, i])
                orders_all.append(n)
                colors_all.append(colors[i])
                markers.append(marker_list[i])
                markerfillstyles.append(markerfillstyle_2bf)
                n_bodies.append('2')
                if n > 2:  # Has 3-body forces
                    c.append(c3[:, i])
                    orders_all.append(n)
                    colors_all.append(colors[i])
                    markers.append(marker_list[i])
                    markerfillstyles.append(markerfillstyle_3bf)
                    n_bodies.append('3')
            c = np.array(c).T
            orders_all = np.array(orders_all)
            print(f'Reseting orders to be {orders_all}')
            y = gm.partials(c, ratio_vals, ref2, orders_all)

            self.y = y
            self.orders = orders_all
            self.ref = ref2
        elif body == 'NN-only':
            self.y = y2
            self.orders = orders
            self.ref = ref2
            colors_all = colors
            markerfillstyles = [markerfillstyle_2bf] * len(orders)
            n_bodies = ['2'] * len(orders)
            markers = marker_list
        elif body == 'NN+3N':
            self.y = y3
            self.orders = orders
            self.ref = ref2
            colors_all = colors
            markerfillstyles = [markerfillstyle_2bf] * len(orders)
            n_bodies = ['2+3'] * len(orders)
            markers = marker_list
        elif body == '3N':
            self.y = y3 - y2
            self.orders = orders
            self.ref = ref3
            colors_all = colors
            markerfillstyles = [markerfillstyle_3bf] * len(orders)
            n_bodies = ['3'] * len(orders)
            markers = marker_list
        else:
            raise ValueError('body not in allowed values')

        self.train = train
        self.valid = valid
        self.X_train = X[train]
        self.X_valid = X[valid]

        self.y2 = y2
        if body != '3N':
            self.y2_train = y2[train]
            self.y2_valid = y2[valid]
        else:
            self.y2_train = None
            self.y2_train = None

        self.y3 = y3
        if body != 'NN-only':
            self.y3_train = y3[train]
            self.y3_valid = y3[valid]
        else:
            self.y3_train = None
            self.y3_train = None

        self.y_train = self.y[train]
        self.y_valid = self.y[valid]

        self.n_bodies = n_bodies

        self.ratio = ratio
        # self.ref = ref
        self.ref2 = ref2
        self.ref3 = ref3
        self.excluded = excluded
        if excluded is None:
            excluded_mask = np.ones_like(self.orders, dtype=bool)
        else:
            excluded_mask = ~np.isin(self.orders, excluded)
        self.excluded_mask = excluded_mask
        self.orders_not_excluded = self.orders[excluded_mask]

        if excluded is None:
            excluded_mask_original = np.ones_like(orders, dtype=bool)
        else:
            excluded_mask_original = ~np.isin(orders, excluded)
        self.excluded_mask_original = excluded_mask_original

        colors_all = np.atleast_1d(colors_all)
        self.colors_not_excluded = colors_all[excluded_mask]
        self.colors = colors_all

        self.colors_original = colors_original = np.atleast_1d(colors_original)
        self.colors_original_not_excluded = colors_original[excluded_mask_original]

        self.orders_original_not_excluded = self.orders_original[excluded_mask_original]

        self.markers = markers = np.atleast_1d(markers)
        self.markers_not_excluded = markers[excluded_mask]

        self.markerfillstyles = markerfillstyles = np.atleast_1d(markerfillstyles)
        self.markerfillstyles_not_excluded = markerfillstyles[excluded_mask]

        self.kwargs = kwargs

    def compute_coefficients(self, show_excluded=False, **kwargs):
        ratio = self.ratio(self.X, **kwargs)
        c = gm.coefficients(self.y, ratio, self.ref, self.orders)
        if not show_excluded:
            c = c[:, self.excluded_mask]
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
    density : ndarray
    system : str
        The physical system to consider. Can be 'neutron', 'symmetric', or 'difference'. Affects how to convert
        between kf and density, and also the way that files are named.
    fit_n2lo : str
        The fit number for the NN+3N N2LO potential. Used for naming files.
    fit_n3lo : str
        The fit number for the NN+3N N3LO potential. Used for naming files.
    Lambda : int
        The Lambda regulator for the potential. Used for naming files.
    body : str
        Either 'NN-only' or 'NN+3N'
    savefigs : bool, optional
        Whether to save figures when plot_* is called. Defaults to `False`

    Other Parameters
    ----------------
    %(ConvergenceAnalysis.other_parameters)s
    """

    system_strings = dict(
        neutron='neutron',
        symmetric='symmetric',
        difference='difference',
    )
    system_strings_short = dict(
        neutron='n',
        symmetric='s',
        difference='d',
    )
    system_math_strings = dict(
        neutron='E/N',
        symmetric='E/A',
        difference='S_2',
    )
    ratio_map = dict(
        kf=ratio_kf
    )

    MD_label = r'\mathrm{D}_{\mathrm{MD}}^2'
    PC_label = r'\mathrm{D}_{\mathrm{PC}}'

    def __init__(self, X, y2, y3, orders, train, valid, ref2, ref3, ratio, density, *, system='neutron',
                 fit_n2lo=None, fit_n3lo=None, Lambda=None, body=None, savefigs=False,
                 fig_path='new_figures', **kwargs):

        self.ratio_str = ratio
        ratio = self.ratio_map[ratio]

        color_list = ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples', 'Greys']
        cmaps = [plt.get_cmap(name) for name in color_list[:len(orders)]]
        colors = [cmap(0.55 - 0.1 * (i == 0)) for i, cmap in enumerate(cmaps)]

        body_vals = ['NN-only', 'NN+3N', '3N', 'Appended']
        if body not in body_vals:
            raise ValueError(f'body must be in {body_vals}')

        # TODO: allow `excluded` to work properly in plots, etc.
        super().__init__(
            X, y2, y3, orders, train, valid, ref2, ref3, ratio, body=body, colors=colors, **kwargs)
        self.system = system
        self.fit_n2lo = fit_n2lo
        self.fit_n3lo = fit_n3lo
        self.Lambda = Lambda
        self.body = body
        self.savefigs = savefigs
        self.fig_path = fig_path
        self.system_math_string = self.system_math_strings[system]
        self.density = density
        self.df_joint = None
        self.df_breakdown = None
        self.df_ls = None
        self.breakdown = None
        self.breakdown_min, self.breakdown_max, self.breakdown_num = None, None, None
        self.ls_min, self.ls_max, self.ls_num = None, None, None
        self._breakdown_map = None
        self._ls_map = None
        self.ls = None
        self.max_idx = None
        self.logprior = None

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

    def setup_posteriors(self, max_idx, breakdown_min, breakdown_max, breakdown_num, ls_min, ls_max, ls_num,
                         logprior=None, max_idx_labels=None):
        R"""Computes and stores the values for the breakdown and length scale posteriors.

        This must be run before running functions that depend on these posteriors.

        Parameters
        ----------
        max_idx : List[int], int
            All orders up to self.orders[:max_idx+1] are kept and used to compute posteriors. If a list is provided,
            then the posterior is computed for each of the max_indices in the list.
        breakdown_min : float
            The minimum value for the breakdown scale. Will be used to compute
            `np.linspace(breakdown_min, breakdown_max, breakdown_num)`.
        breakdown_max : float
            The maximum value for the breakdown scale. Will be used to compute
            `np.linspace(breakdown_min, breakdown_max, breakdown_num)`.
        breakdown_num : int
            The number of breakdown scale values to use in the posterior. Will be used to compute
            `np.linspace(breakdown_min, breakdown_max, breakdown_num)`.
        ls_min : float
            The minimum value for the length scale. Will be used to compute
            `np.linspace(ls_min, ls_max, ls_num)`. if `ls_min`, `ls_max`, and `ls_num` are all `None`, then
            the MAP value of the length scale will be used for the breakdown posterior. No length scale posterior
            will be computed in this case.
        ls_max : float
            The maximum value for the length scale. Will be used to compute
            `np.linspace(ls_min, ls_max, ls_num)`. if `ls_min`, `ls_max`, and `ls_num` are all `None`, then
            the MAP value of the length scale will be used for the breakdown posterior. No length scale posterior
            will be computed in this case.
        ls_num : int
            The number of length scales to use in the posterior. Will be used to compute
            `np.linspace(ls_min, ls_max, ls_num)`. if `ls_min`, `ls_max`, and `ls_num` are all `None`, then
            the MAP value of the length scale will be used for the breakdown posterior. No length scale posterior
            will be computed in this case.
        logprior : ndarray, optional, shape = (ls_num, breakdown_num)
            The prior pr(breakdown, ls). If `None`, then a flat prior is used.
        Returns
        -------

        """
        dfs_breakdown = []
        dfs_ls = []
        dfs_joint = []
        self.breakdown_min, self.breakdown_max, self.breakdown_num = breakdown_min, breakdown_max, breakdown_num
        self.ls_min, self.ls_max, self.ls_num = ls_min, ls_max, ls_num
        breakdown = np.linspace(breakdown_min, breakdown_max, breakdown_num)
        if ls_min is None and ls_max is None and ls_num is None:
            ls = None
        else:
            ls = np.linspace(ls_min, ls_max, ls_num)
        breakdown_maps = []
        ls_maps = []
        max_idx = np.atleast_1d(max_idx)
        if max_idx_labels is None:
            max_idx_labels = max_idx
        for idx, idx_label in zip(max_idx, max_idx_labels):
            joint_pdf, breakdown_pdf, ls_pdf = self.compute_breakdown_ls_posterior(
                breakdown, ls, max_idx=idx, logprior=logprior)

            df_breakdown = pd.DataFrame(np.array([breakdown, breakdown_pdf]).T, columns=[r'$\Lambda_b$ (MeV)', 'pdf'])
            df_breakdown['Order'] = fr'N$^{idx_label}$LO'
            df_breakdown['Order Index'] = idx
            df_breakdown['system'] = fr'${self.system_math_string}$'
            df_breakdown['Body'] = self.body
            dfs_breakdown.append(df_breakdown)

            if ls is not None:
                df_ls = pd.DataFrame(np.array([ls, ls_pdf]).T, columns=[r'$\ell$ (fm$^{-1}$)', 'pdf'])
                df_ls['Order'] = fr'N$^{idx_label}$LO'
                df_ls['Order Index'] = idx
                df_ls['system'] = fr'${self.system_math_string}$'
                df_ls['Body'] = self.body
                dfs_ls.append(df_ls)

            X = gm.cartesian(ls, breakdown)
            df_joint = pd.DataFrame(X, columns=[r'$\ell$ (fm$^{-1}$)', r'$\Lambda_b$ (MeV)'])
            df_joint['pdf'] = joint_pdf.ravel()
            df_joint['Order'] = fr'N$^{idx_label}$LO'
            df_joint['Order Index'] = idx
            df_joint['system'] = fr'${self.system_math_string}$'
            df_joint['Body'] = self.body
            dfs_joint.append(df_joint)

            map_idx = np.argmax(joint_pdf)
            map_idx = np.unravel_index(map_idx, joint_pdf.shape)
            breakdown_maps.append(breakdown[map_idx[1]])
            if ls is not None:
                ls_maps.append(ls[map_idx[0]])
        df_breakdown = pd.concat(dfs_breakdown, ignore_index=True)
        df_ls = None
        if ls is not None:
            df_ls = pd.concat(dfs_ls, ignore_index=True)
        df_joint = pd.concat(dfs_joint, ignore_index=True)
        self.breakdown = breakdown
        self.ls = ls
        self.logprior = logprior
        self.max_idx = max_idx
        self.max_idx_labels = max_idx_labels
        self.df_joint = df_joint
        self.df_breakdown = df_breakdown
        self.df_ls = df_ls
        self._breakdown_map = breakdown_maps
        self._ls_map = ls_maps
        return df_joint, df_breakdown, df_ls

    @property
    def breakdown_map(self):
        return self._breakdown_map

    @property
    def ls_map(self):
        return self._ls_map

    def compute_underlying_graphical_diagnostic(self, breakdown, show_excluded=False):
        coeffs = coeffs_not_excluded = self.compute_coefficients(
            breakdown=breakdown, show_excluded=show_excluded
        )
        colors = self.colors
        markerfillstyles = self.markerfillstyles
        markers = self.markers
        if not show_excluded:
            colors = self.colors_not_excluded
            markerfillstyles = self.markerfillstyles_not_excluded
            markers = self.markers_not_excluded
            coeffs_not_excluded = self.compute_coefficients(breakdown=breakdown, show_excluded=False)

        process = gm.ConjugateGaussianProcess(**self.kwargs)
        process.fit(self.X_train, coeffs_not_excluded[self.train])  # in either case, only fit to non-excluded coeffs
        mean = process.mean(self.X_valid)
        cov = process.cov(self.X_valid)
        # But it may be useful to visualize the diagnostics off all coefficients
        graph = gm.GraphicalDiagnostic(
            coeffs[self.valid], mean, cov, colors=colors, gray=gray, black=softblack,
            markerfillstyles=markerfillstyles, markers=markers
        )
        return graph

    def compute_breakdown_ls_posterior(self, breakdown, ls, max_idx=None, logprior=None):
        # orders = self.orders[:max_idx + 1]
        orders = self.orders
        model = gm.TruncationGP(ref=self.ref, ratio=self.ratio, excluded=self.excluded, **self.kwargs)
        X = self.X_train
        data = self.y_train
        joint_pdf, Lb_pdf, ls_pdf = compute_2d_posterior(
            model, X, data, orders, breakdown, ls, logprior=logprior, max_idx=max_idx,
        )
        return joint_pdf, Lb_pdf, ls_pdf

    def compute_best_length_scale_for_breakdown(self, breakdown, max_idx):
        ord = rf'N$^{max_idx}$LO'
        df_best = self.df_joint[
            (self.df_joint[r'$\Lambda_b$ (MeV)'] == breakdown) &
            (self.df_joint['Order'] == ord)
        ]
        ls_max_idx = df_best['pdf'].idxmax()
        return df_best.loc[ls_max_idx][r'$\ell$ (fm$^{-1}$)']

    def order_index(self, order):
        return np.squeeze(np.argwhere(self.orders == order))

    def setup_and_fit_truncation_process(self, breakdown):
        model = gm.TruncationGP(
            ratio=self.ratio, ref=self.ref, excluded=self.excluded,
            ratio_kws=dict(breakdown=breakdown), **self.kwargs
        )
        # Only update hyperparameters based on train
        model.fit(self.X_train, y=self.y_train, orders=self.orders)
        return model

    def compute_minimum(self, order, n_samples, breakdown=None, X=None, nugget=0, cond=None):
        if X is None:
            X = self.X
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
        if cond is None:
            cond = self.train
        x = X.ravel()
        # ord = self.orders == order
        orders = self.orders_original
        # colors = self.colors_original

        if self.body == 'NN-only':
            y = self.y2
        elif self.body == 'NN+3N':
            y = self.y3
        elif self.body == 'Appended':
            y = self.y3
        elif self.body == '3N':
            y = self.y3
        else:
            raise ValueError('body not in allowed values')

        ord = np.squeeze(np.argwhere(orders == order))

        if ord.ndim > 0:
            raise ValueError('Found multiple orders that match order')

        model = gm.TruncationGP(
            ratio=self.ratio, ref=self.ref, excluded=self.excluded,
            ratio_kws=dict(breakdown=breakdown), **self.kwargs
        )
        # Only update hyperparameters based on train
        model.fit(self.X_train, y=self.y_train, orders=self.orders)
        print(model.coeffs_process.kernel_)
        # But then condition on `cond` X, y points to get a good interpolant
        pred, cov = model.predict(X, order=order, return_cov=True, Xc=self.X[cond], y=y[cond, ord], kind='both')
        # pred, cov = model.predict(X, order=order, return_cov=True, kind='both')
        # pred += self.y[:, ord]
        # cov += np.diag(cov) * nugget * np.eye(cov.shape[0])
        x_min, y_min = minimum_samples(pred, (cov + nugget * np.eye(cov.shape[0])), n=n_samples, x=x)

        is_endpoint = x_min == X[-1].ravel()
        x_min = x_min[~is_endpoint]
        y_min = y_min[~is_endpoint]

        # Don't interpolate
        # min_idx = np.argmin(self.y[:, ord])
        # x_min_no_trunc, y_min_no_trunc = self.X.ravel()[min_idx], self.y[min_idx][ord]

        # Do interpolate
        min_idx = np.argmin(pred)
        x_min_no_trunc, y_min_no_trunc = X.ravel()[min_idx], pred[min_idx]
        return x_min_no_trunc, y_min_no_trunc, x_min, y_min, pred, cov

    def figure_name(self, prefix, breakdown=None, ls=None, max_idx=None, include_system=True):
        body = self.body
        fit_n2lo = self.fit_n2lo
        fit_n3lo = self.fit_n3lo
        Lambda = self.Lambda
        ref = self.ref
        if not include_system:
            system = 'x'
        else:
            system = self.system_strings_short[self.system]

        full_name = prefix + f'sys-{system}_{body}'
        if body == 'NN+3N' or body == '3N':
            full_name += f'_fit-{fit_n2lo}-{fit_n3lo}'
        else:
            full_name += f'_fit-0-0'
        full_name += f'_Lamb-{Lambda:.0f}_Q-{self.ratio_str}'

        if isinstance(breakdown, tuple):
            full_name += f'_Lb-{breakdown[0]:.0f}-{breakdown[1]:.0f}-{breakdown[2]:.0f}'
        elif breakdown is not None:
            full_name += f'_Lb-{breakdown:.0f}'
        else:
            full_name += f'_Lb-x'

        if isinstance(ls, tuple):
            full_name += f'_ls-{ls[0]:.0f}-{ls[1]:.0f}-{ls[2]:.0f}'
        elif ls is not None:
            full_name += f'_ls-{ls:.0f}'
        else:
            full_name += f'_ls-x'
        full_name += f'_ref-{ref:.0f}'
        if max_idx is not None:
            full_name += f'_midx-{max_idx}'
        else:
            full_name += f'_midx-x'

        center = self.kwargs.get('center', 0)
        disp = self.kwargs.get('disp', 1)
        df = self.kwargs.get('df', 1)
        scale = self.kwargs.get('scale', 1)
        full_name += f'_hyp-{center}-{disp}-{df}-{scale}'

        full_name = join(self.fig_path, full_name)
        return full_name

    def model_info(self, breakdown=None, ls=None, max_idx=None):
        if breakdown is None:
            breakdown = np.NaN
        if ls is None:
            ls = np.NaN
        if max_idx is None:
            max_idx = np.NaN
        info = dict(
            body=self.body,
            fit_n2lo=self.fit_n2lo,
            fit_n3lo=self.fit_n3lo,
            Lambda=self.Lambda,
            ref=self.ref,
            center=self.kwargs.get('center', 0),
            disp=self.kwargs.get('disp', 1),
            df=self.kwargs.get('df', 1),
            scale=self.kwargs.get('scale', 1),
            breakdown=breakdown,
            ls=ls,
            max_idx=max_idx,
        )
        return info

    def compute_y_label(self):
        if self.system == 'neutron':
            y_label = fr'Energy per Neutron '
        elif self.system == 'symmetric':
            y_label = 'Energy per Particle '
        elif self.system == 'difference':
            y_label = 'Symmetry Energy '
        else:
            raise ValueError('system has wrong value')
        y_label += fr'${self.system_math_strings[self.system]}$'
        return y_label

    def setup_ticks(self, ax, is_density_primary, train, valid, show_2nd_axis=True):
        d_label = r'Density $n$ (fm$^{-3}$)'
        kf_label = r'Fermi Momentum $k_\mathrm{F}$ (fm$^{-1}$)'
        # ax.set_xticks(x_ticks)
        # ax2.set_xticks(x_ticks)

        if is_density_primary:
            x_label = d_label
            x = self.density
            x_ticks = x[train]

            if show_2nd_axis:
                x_label2 = kf_label
                x_ticks2 = self.compute_momentum(x_ticks)

            # ax.set_xlabel(d_label)
            # ax.set_xticks(x_ticks)
            # ax.set_xticks(self.density[valid], minor=True)
            #
            # ax2.plot(x_ticks, ax.get_yticks().mean() * np.ones_like(x_ticks), ls='')
            # ax2.set_xlabel(kf_label)
            # ax2.set_xticklabels(self.compute_momentum(x_ticks))
        else:
            x_label = kf_label
            x = self.X.ravel()
            x_ticks = x[train]

            if show_2nd_axis:
                x_label2 = d_label
                x_ticks2 = self.compute_density(x_ticks)

            # ax.set_xlabel(kf_label)
            # x_ticks = self.X[train].ravel()
            # ax.set_xticks(x_ticks)
            # ax.set_xticks(self.X[valid].ravel(), minor=True)
            #
            # ax2.plot(x_ticks, ax.get_yticks().mean() * np.ones_like(x_ticks), ls='')
            # ax2.set_xlabel(d_label)
            # ax2.set_xticks(x_ticks)
            # ax2.set_xticklabels(self.compute_density(x_ticks))

        ax.set_xlabel(x_label)
        ax.set_xticks(x_ticks)
        ax.set_xticks(x[valid], minor=True)
        ax.tick_params(right=True)

        y_label = self.compute_y_label()
        ax.set_ylabel(y_label)

        if show_2nd_axis:
            ax2 = ax.twiny()
            # Plot invisible line to get ticks right
            ax2.plot(x_ticks, ax.get_yticks().mean() * np.ones_like(x_ticks), ls='')
            ax2.set_xlabel(x_label2)
            ax2.set_xticks(x_ticks)
            ax2.set_xticks(x[valid], minor=True)
            ax2.set_xticklabels([f'{tick:0.2f}' for tick in x_ticks2])
            return ax, ax2
        return ax

    def plot_coefficients(self, breakdown=None, ax=None, show_process=False, savefig=None, return_info=False,
                          show_excluded=False, show_2nd_axis=True):
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')

        if ax is None:
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
        kf = self.X.ravel()
        density = self.density
        train = self.train

        if show_process:
            coeffs_not_excluded = self.compute_coefficients(breakdown=breakdown, show_excluded=False)
            model = gm.ConjugateGaussianProcess(**self.kwargs)
            model.fit(self.X_train, coeffs_not_excluded[train])
            print(model.kernel_)
            if show_excluded:
                model_all = gm.ConjugateGaussianProcess(**self.kwargs)
                coeffs_all = self.compute_coefficients(breakdown=breakdown, show_excluded=True)
                model_all.fit(self.X_train, coeffs_all[train])
                pred, std = model_all.predict(self.X, return_std=True)
            else:
                pred, std = model.predict(self.X, return_std=True)
            mu = model.center_
            cbar = np.sqrt(model.cbar_sq_mean_)
            ax.axhline(mu, 0, 1, c='k', zorder=0)
            ax.axhline(2*cbar, 0, 1, c=gray, zorder=0)
            ax.axhline(-2*cbar, 0, 1, c=gray, zorder=0)

        coeffs = self.compute_coefficients(breakdown=breakdown, show_excluded=show_excluded)
        colors = self.colors
        orders = self.orders
        markers = self.markers
        markerfillstyles = self.markerfillstyles
        if not show_excluded:
            colors = self.colors_not_excluded
            orders = self.orders_not_excluded
            markers = self.markers_not_excluded
            markerfillstyles = self.markerfillstyles_not_excluded
        light_colors = [lighten_color(c, 0.5) for c in colors]

        is_density_primary = True
        if is_density_primary:
            x = density
        else:
            x = kf

        for i, n in enumerate(orders):
            z = i / 20
            ax.plot(
                x, coeffs[:, i], c=colors[i], label=fr'$c_{{{n}}}^{{({self.n_bodies[i]})}}$', zorder=z,
                markevery=train, marker=markers[i], fillstyle=markerfillstyles[i])
            # ax.plot(x[train], coeffs[train, i], marker=markers[i], ls='', c=colors[i], zorder=z,
            #         fillstyle=markerfillstyles[i])
            if show_process:
                ax.plot(x, pred[:, i], c=colors[i], zorder=z, ls='--')
                ax.fill_between(
                    x, pred[:, i] + 2*std, pred[:, i] - 2*std, zorder=z,
                    lw=0.5, alpha=1, facecolor=light_colors[i], edgecolor=colors[i]
                )
        ax.axhline(0, 0, 1, ls='--', c=gray, zorder=-1)
        # ax2 = ax.twiny()
        # ax2.plot(d, np.zeros_like(d), ls='', c=gray, zorder=-1)  # Dummy data to set up ticks
        # ax2.set_xlabel(r'Density $n$ (fm$^{-3}$)')

        # y_label = self.compute_y_label()
        # ax.set_ylabel(y_label)
        # ax.set_xlabel(r'Fermi Momentum $k_\mathrm{F}$ (fm$^{-1}$)')
        # ax.set_xticks(self.X_valid.ravel(), minor=True)
        if len(orders) > 4:
            ax.legend(ncol=3)
        else:
            ax.legend(ncol=2)

        self.setup_ticks(ax, is_density_primary, train=train, valid=self.valid, show_2nd_axis=show_2nd_axis)

        if savefig is None:
            savefig = self.savefigs

        if savefig:
            fig = plt.gcf()
            name = self.figure_name('coeffs', breakdown=breakdown)
            fig.savefig(name)

            if return_info:
                info = self.model_info(breakdown=breakdown)
                name = path.relpath(name, self.fig_path)
                info['name'] = name
                return ax, info

        return ax

    def plot_observables(self, breakdown=None, ax=None, show_process=False, savefig=None, return_info=False,
                         show_excluded=False, show_2nd_axis=True):
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')

        if ax is None:
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.margins(x=0.)
        kf = self.X.ravel()

        is_density_primary = True
        if is_density_primary:
            x = self.density
        else:
            x = kf

        if show_process:
            model = gm.TruncationGP(
                ratio=self.ratio, ref=self.ref, excluded=self.excluded,
                ratio_kws=dict(breakdown=breakdown), **self.kwargs
            )
            model.fit(self.X_train, y=self.y_train, orders=self.orders)

        if self.body == 'NN-only':
            y = self.y2
        elif self.body == 'NN+3N':
            y = self.y3
        elif self.body == 'Appended':
            y = self.y3
        elif self.body == '3N':
            y = self.y3
        else:
            raise ValueError('body not in allowed values')

        orders = self.orders_original
        colors = self.colors_original
        if not show_excluded:
            # coeffs = coeffs[:, self.excluded_mask]
            colors = self.colors_original_not_excluded
            orders = self.orders_original_not_excluded

        light_colors = [lighten_color(c, 0.5) for c in colors]

        for i, n in enumerate(orders):
            z = i / 20
            if n not in self.orders_not_excluded and not show_excluded:
                # Don't plot orders if we've excluded them
                continue
            order_label = n if n in [0, 1] else n - 1
            ax.plot(x, y[:, i], c=colors[i], label=fr'N$^{order_label}$LO', zorder=z)
            # ax.plot(kf[train], self.y[train, i], marker='o', ls='', c=colors[i], zorder=z)
            if show_process:
                _, std = model.predict(self.X, order=n, return_std=True, kind='trunc')
                if self.body == 'Appended':
                    n_3bf = n if n >= 3 else 3  # 3-body forces don't enter until N3LO
                    _, std_3bf = model.predict(self.X, order=n_3bf, return_std=True, kind='trunc')
                    try:
                        ref3_vals = self.ref3(self.X)
                    except TypeError:
                        ref3_vals = self.ref3
                    try:
                        ref2_vals = self.ref2(self.X)
                    except TypeError:
                        ref2_vals = self.ref2
                    # For appended, the standard reference is the 2-body one. So swap for the 3-body ref
                    std_3bf *= ref3_vals / ref2_vals
                    std = np.sqrt(std**2 + std_3bf**2)
                ax.plot(x, y[:, i], c=colors[i], zorder=z, ls='--')
                ax.fill_between(
                    x, y[:, i] + 2*std, y[:, i] - 2*std, zorder=z,
                    lw=0.5, alpha=1, facecolor=light_colors[i], edgecolor=colors[i]
                )

        # ax2.plot(d, self.y[:, 0], ls='', c=gray, zorder=-1)  # Dummy data to set up ticks
        # ax.axhline(0, 0, 1, ls='--', c=gray, zorder=-1)

        # if self.system == 'neutron':
        #     y_label = fr'Energy per Neutron '
        # elif self.system == 'symmetric':
        #     y_label = 'Energy per Particle '
        # elif self.system == 'difference':
        #     y_label = 'Symmetry Energy '
        # else:
        #     raise ValueError('system has wrong value')
        #
        # y_label += fr'${self.system_math_strings[self.system]}$'
        # y_label = self.compute_y_label()
        # ax.set_ylabel(y_label)
        # ax.set_xlabel(r'Fermi Momentum $k_\mathrm{F}$ (fm$^{-1}$)')
        # ax.set_xticks(self.X_valid.ravel(), minor=True)
        ax.legend()

        ax.margins(x=0.)
        from matplotlib.ticker import MultipleLocator
        # if self.system == 'neutron':
        #     kf_ticks = np.array([1.2, 1.4, 1.6, 1.8])
        # elif self.system == 'symmetric':
        #     kf_ticks = np.array([1., 1.2, 1.4])
        # else:
        #     kf_ticks = np.array([1., 1.2, 1.4])
        # ax.set_xticks(kf_ticks)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))

        # ax2 = ax.twiny()
        # ax2.margins(x=0.)
        ax.set_xlim(x[0], x[-1])

        axes = self.setup_ticks(
            ax, is_density_primary, train=self.train, valid=self.valid, show_2nd_axis=show_2nd_axis)
        if show_2nd_axis:
            axes[-1].set_xlim(x[0], x[-1])

        if self.system == 'symmetric':
            self.plot_empirical_saturation(ax, is_density_primary=is_density_primary)

        if savefig is None:
            savefig = self.savefigs

        if savefig:
            fig = plt.gcf()
            name = self.figure_name('obs_', breakdown=breakdown)
            fig.savefig(name)

            if return_info:
                info = self.model_info(breakdown=breakdown)
                info['name'] = path.relpath(name, self.fig_path)
                return ax, info
        return ax

    def plot_joint_breakdown_ls(self, max_idx, return_info=False):
        system_str = fr'${self.system_math_string}$'
        order_str = fr'N$^{max_idx}$LO'
        fig = joint2dplot(self.df_ls, self.df_breakdown, self.df_joint, system=system_str,
                          order=order_str, data_str=self.system_math_string)

        breakdown = (self.breakdown_min, self.breakdown_max, self.breakdown_num)
        ls = (self.ls_min, self.ls_max, self.ls_num)
        if self.savefigs:
            name = self.figure_name('ls-Lb-2d_', breakdown=breakdown, ls=ls, max_idx=max_idx)
            fig.savefig(name)

            if return_info:
                info = self.model_info(max_idx=max_idx)
                info['name'] = path.relpath(name, self.fig_path)
                return fig, info
        return fig

    def plot_md_squared(self, breakdown=None, ax=None, savefig=None, return_info=False):
        R"""Plots the squared Mahalanobis distance.

        Parameters
        ----------
        breakdown : float, optional
            The value for the breakdown scale to use in the diagnostics. If `None`, then its MAP value is used.
        ax : matplotlib.axes.Axes, optional
            The axis on which to draw the coefficient plots and diagnostics
        savefig : bool, optional
            Whether to save the figure. If `None`, this is taken from `self.savefigs`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(1, 3.2))
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')
        graph = self.compute_underlying_graphical_diagnostic(breakdown=breakdown)
        obs = self.system_math_string
        ax = graph.md_squared(type='box', trim=True, title=None, xlabel=rf'${self.MD_label}({obs})$', ax=ax)

        if savefig is None:
            savefig = self.savefigs
        if savefig:
            fig = plt.gcf()
            name = self.figure_name('md_under_', breakdown=breakdown)
            fig.savefig(name)

            if return_info:
                info = self.model_info(breakdown=breakdown)
                info['name'] = path.relpath(name, self.fig_path)
                return ax, info
        return ax

    def plot_pchol(self, breakdown=None, ax=None, savefig=None, return_info=False):
        R"""Plots the pivoted Cholesky diagnostic.

        Parameters
        ----------
        breakdown : float, optional
            The value for the breakdown scale to use in the diagnostic. If `None`, then its MAP value is used.
        ax : matplotlib.axes.Axes, optional
            The axis on which to draw the coefficient plots and diagnostics
        savefig : bool, optional
            Whether to save the figure. If `None`, this is taken from `self.savefigs`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')
        graph = self.compute_underlying_graphical_diagnostic(breakdown=breakdown)
        obs = self.system_math_string
        with plt.rc_context({"text.usetex": True, "text.latex.preview": True}):
            ax = graph.pivoted_cholesky_errors(ax=ax, title=None)
            ax.text(0.04, 0.967, rf'${self.PC_label}({obs})$', bbox=text_bbox, transform=ax.transAxes, va='top',
                    ha='left')
            fig = plt.gcf()

            if savefig is None:
                savefig = self.savefigs

            if savefig:
                name = self.figure_name('pc_under_', breakdown=breakdown)
                fig.savefig(name)

                if return_info:
                    info = self.model_info(breakdown=breakdown)
                    info['name'] = path.relpath(name, self.fig_path)
                    return ax, info
        return ax

    def plot_coeff_diagnostics(self, breakdown=None, fig=None, savefig=None, return_info=False):
        R"""Plots coefficients, the squared Mahalanobis distance, and the pivoted Cholesky diagnostic.

        Parameters
        ----------
        breakdown : float, optional
            The value for the breakdown scale to use in the diagnostics. If `None`, then its MAP value is used.
        fig : matplotlib.figure.Figure, optional
            The Figure on which to draw the coefficient plots and diagnostics
        savefig : bool, optional
            Whether to save the figure. If `None`, this is taken from `self.savefigs`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        """
        if fig is None:
            fig = plt.figure(figsize=(8, 3.2), constrained_layout=True)
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')
        spec = fig.add_gridspec(nrows=1, ncols=7)
        ax_cs = fig.add_subplot(spec[:, :3])
        ax_md = fig.add_subplot(spec[:, 3])
        ax_pc = fig.add_subplot(spec[:, 4:])
        show_2nd_axis = self.system != self.system_strings['difference']
        self.plot_coefficients(
            breakdown=breakdown, ax=ax_cs, show_process=True, savefig=False, show_2nd_axis=show_2nd_axis)
        self.plot_md_squared(breakdown=breakdown, ax=ax_md, savefig=False)
        self.plot_pchol(breakdown=breakdown, ax=ax_pc, savefig=False)

        if savefig is None:
            savefig = self.savefigs
        if savefig:
            name = self.figure_name('cn_diags_', breakdown=breakdown)
            # fig.savefig(name, metadata={'hi': [1, 2, 3], 'wtf': 7})
            fig.savefig(name)

            if return_info:
                info = self.model_info(breakdown=breakdown)
                info['name'] = path.relpath(name, self.fig_path)
                return fig, info
        return fig

    def plot_empirical_saturation(self, ax=None, is_density_primary=True):
        from matplotlib.patches import Rectangle
        # From Drischler 2018 arXiv:1710.08220
        n0 = 0.164
        n0_std = 0.007
        y0 = -15.86
        y0_std = np.sqrt(0.37 ** 2 + 0.2 ** 2)
        left = n0 - n0_std
        right = n0 + n0_std
        if not is_density_primary:
            left = self.compute_momentum(left)
            right = self.compute_momentum(right)
        rect = Rectangle(
            (left, y0 - y0_std), width=right - left, height=2 * y0_std,
            facecolor='lightgray', edgecolor='gray', alpha=0.4, zorder=9,
        )
        ax.add_patch(rect)
        return ax

    def plot_saturation(self, breakdown=None, order=4, ax=None, savefig=None, color=None, nugget=0, X=None,
                        cond=None, n_samples=1000, **kwargs):
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')
        if ax is None:
            ax = plt.gca()
        if X is None:
            X = self.X
        x_min_no_trunc, y_min_no_trunc, x_min, y_min, pred, cov = self.compute_minimum(
            order=order, n_samples=n_samples, breakdown=breakdown, X=X, nugget=nugget, cond=cond
        )

        if cond is None:
            cond = slice(None, None)
        # ord_idx = self.order_index(order)
        ord_idx = np.squeeze(np.argwhere(self.orders_original == order))
        approx_xlim = x_min.min() - 0.03, x_min.max() + 0.03
        approx_xlim_mask = (self.X[cond].ravel() >= approx_xlim[0]) & (self.X[cond].ravel() <= approx_xlim[1])

        is_density_primary = True

        if is_density_primary:
            x_min_no_trunc = self.compute_density(x_min_no_trunc)
            x_min = self.compute_density(x_min)
            x_all = self.compute_density(X.ravel())
        else:
            x_all = X.ravel()

        if color is None:
            color = self.colors_original[ord_idx]
        light_color = lighten_color(color)
        # TODO: Add scatter plots
        # compute z-scores from all EDFs?
        stdv = np.sqrt(np.diag(cov))
        from matplotlib.collections import LineCollection
        # ax.fill_between(X.ravel(), pred+stdv, pred-stdv, color=color, zorder=0, alpha=0.5)
        # ax.plot(X.ravel(), pred, c=color)
        col = LineCollection([
            np.column_stack((x_all, pred)),
            np.column_stack((x_all, pred+2*stdv)),
            np.column_stack((x_all, pred-2*stdv))
        ], colors=[color, color, color], linewidths=[1.2, 0.7, 0.7], linestyles=['-', '-', '-'])
        ax.add_collection(col, autolim=False)
        ellipse = confidence_ellipse(
            x_min, y_min, ax=ax, n_std=2, facecolor=light_color,
            zorder=0, show_scatter=True, **kwargs
        )
        # ax.plot(x_min_no_trunc, y_min_no_trunc, marker='x', ls='', markerfacecolor=color,
        #         markeredgecolor='k', markeredgewidth=0.5, label='True', zorder=10)
        ax.scatter(x_min_no_trunc, y_min_no_trunc, marker='X', facecolor=color,
                   edgecolors='k', label=fr'min($y_{order}$)', zorder=10)
        # ax.scatter(x_min, y_min, marker='X', facecolor=color,
        #            edgecolors='k', label=fr'min($y_{order}$)', zorder=10)


        if self.body == 'NN-only':
            y = self.y2
        elif self.body == 'NN+3N':
            y = self.y3
        elif self.body == 'Appended':
            y = self.y3
        elif self.body == '3N':
            y = self.y3
        else:
            raise ValueError('body not in allowed values')

        if is_density_primary:
            ax.plot(self.density[cond][approx_xlim_mask], y[cond, ord_idx][approx_xlim_mask],
                    ls='', marker='o', c=color)
            ax.set_xlabel(r'Density $n$ (fm$^{-3}$)')
        else:
            ax.plot(self.X[cond][approx_xlim_mask], y[cond, ord_idx][approx_xlim_mask],
                    ls='', marker='o', c=color)
            ax.set_xlabel(r'Fermi Momentum $k_\mathrm{F}$ (fm$^{-1}$)')
        ax.set_ylabel(r'Energy per Particle $E/A$')
        # kf_ticks = ax.get_xticks()
        # d_ticks = self.compute_momentum(kf_ticks)

        # k_min, k_max = ax.get_xlim()
        # d = self.compute_density(np.array([k_min, k_max]))
        # ax2 = ax.twiny()
        # ax2.plot(d_ticks, np.average(y_min) * np.ones_like(d_ticks), ls='')
        # ax2.set_xticks(d_ticks)
        # is_density_primary = True
        self.plot_empirical_saturation(ax=ax, is_density_primary=is_density_primary)
        if savefig:
            pass
        return ax, ellipse

    def plot_multi_saturation(self, breakdown=None, orders=[3, 4], ax=None, savefig=None,  nugget=0, X=None,
                              cond=None, n_samples=1000, **kwargs):
        if ax is None:
            ax = plt.gca()
        if breakdown is None:
            breakdown = self.breakdown_map[-1]
            print('Using breakdown =', breakdown, 'MeV')
        ellipses = []
        ellipses_labels = []
        for order in orders:
            # idx = self.order_index(order)
            idx = np.squeeze(np.argwhere(self.orders_original == order))
            _, ellipse = self.plot_saturation(
                breakdown=breakdown, order=order, ax=ax, savefig=False, color=self.colors_original[idx],
                nugget=nugget, X=X, cond=cond, n_samples=n_samples, **kwargs)
            ellipses.append(ellipse)
            ellipses_labels.append(rf'$2\sigma(y_{{{order}}}+\delta y_{{{order}}})$')

        ax.margins(x=0)
        handles, labels = ax.get_legend_handles_labels()
        handles = handles + ellipses
        labels = labels + ellipses_labels
        ax.legend(handles, labels)
        fig = plt.gcf()
        # fig.tight_layout()
        if savefig:
            ords = [f'-{order}' for order in orders]
            ords = ''.join(ords)
            name = self.figure_name(f'sat_ellipse_ords{ords}_', breakdown=breakdown)
            fig.savefig(name)
        return ax
