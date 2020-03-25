import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import legend_handler
from matplotlib.legend_handler import HandlerLine2D
import numpy as np


def setup_rc_params(presentation=False):
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = 'k'

    mpl.rcdefaults()  # Set to defaults

    mpl.rc('text', usetex=True)
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'

    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['axes.edgecolor'] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.labelcolor'] = black
    mpl.rcParams['axes.titlesize'] = fontsize

    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['xtick.color'] = black
    mpl.rcParams['ytick.color'] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams['xtick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
    mpl.rcParams['ytick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams['xtick.minor.size'] = 2.4  # Default 2.0
    mpl.rcParams['ytick.minor.size'] = 2.4
    mpl.rcParams['xtick.major.size'] = 3.9  # Default 3.5
    mpl.rcParams['ytick.major.size'] = 3.9

    ppi = 72  # points per inch
    # dpi = 150
    mpl.rcParams['figure.titlesize'] = fontsize
    mpl.rcParams['figure.dpi'] = 150  # To show up reasonably in notebooks
    mpl.rcParams['figure.constrained_layout.use'] = True
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams['figure.constrained_layout.wspace'] = 0.0
    mpl.rcParams['figure.constrained_layout.hspace'] = 0.0
    mpl.rcParams['figure.constrained_layout.h_pad'] = 3. / ppi  # 3 points
    mpl.rcParams['figure.constrained_layout.w_pad'] = 3. / ppi

    mpl.rcParams['legend.title_fontsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
    mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.borderaxespad'] = 0.8
    mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams['hatch.linewidth'] = 0.5

    # bbox = 'tight' can distort the figure size when saved (that's its purpose).
    # mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.04, dpi=350, format='png')
    mpl.rc('savefig', transparent=False, bbox=None, dpi=400, format='png')


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


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', show_scatter=False, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean = np.array([mean_x, mean_y])
    cov = np.cov(x, y)

    if show_scatter:
        from .stats_utils import darken_color
        scat_color = darken_color(facecolor, 0.5)
        ax.plot(x, y, ls='', marker='.', markersize=0.6, color=scat_color)

    return confidence_ellipse_mean_cov(mean, cov, ax, n_std=n_std, facecolor=facecolor, **kwargs)


def confidence_ellipse_mean_cov(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


class OrderBandsHandler(legend_handler.HandlerPolyCollection):

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        light_rect = orig_handle[0]
        dark_rect = orig_handle[1]
        dark_line = orig_handle[2]
        dark_color = dark_line.get_color()
        color = dark_rect.get_facecolor()[0]
        light_color = light_rect.get_facecolor()[0]
        lw = 0.8

        outer_rect = mpatches.Rectangle(
            [0, 0], 1, 1, facecolor='none', edgecolor=dark_color,
            lw=lw
        )
        dark_rect = mpatches.Rectangle(
            [0, 0], 1, 1, facecolor=color, edgecolor='none',
            lw=0
        )
        light_rect = mpatches.Rectangle(
            [0, 0], 1, 1, facecolor=light_color, edgecolor='none',
            lw=0
        )

        patches = []

        factor = 2
        dark_height = 0.4 * height
        #         patches += super().create_artists(legend, light_rect, xdescent, ydescent, width, height, fontsize, trans)
        #         patches += super().create_artists(
        #             legend, dark_rect, xdescent, ydescent-height/(2*factor), width, height/factor, fontsize, trans)
        # print(ydescent, height)
        patches += legend_handler.HandlerPatch().create_artists(
            legend, light_rect, xdescent, ydescent, width, height, fontsize, trans)
        # patches += legend_handler.HandlerPatch().create_artists(
        #     legend, dark_rect, xdescent, ydescent - height / (2 * factor), width, height / factor, fontsize, trans)
        patches += legend_handler.HandlerPatch().create_artists(
            legend, dark_rect, xdescent, ydescent - height / 2 + dark_height/2, width, dark_height, fontsize, trans)

        outer_patches = legend_handler.HandlerPatch().create_artists(
            legend, outer_rect, xdescent, ydescent, width, height, fontsize, trans)
        outer_patches[0].set_linewidth(lw / 2)
        patches += outer_patches

        # line_patches = HandlerLine2D().create_artists(
        #     legend, dark_line, xdescent, ydescent, width, height, fontsize, trans)
        # line_patches[0].set_linewidth(lw)
        # patches += line_patches

        return patches


def compute_filled_handles(colors, light_colors, dark_colors):
    handles = []
    for color, light_color, dark_color in zip(colors, light_colors, dark_colors):
        fill_light = plt.fill_between([], [], [], alpha=1, color=light_color)
        fill_normal = plt.fill_between([], [], [], alpha=1, color=color)
        line_dark = plt.plot([], [], color=dark_color, dash_capstyle='butt', solid_capstyle='butt')[0]

        handles.append((fill_light, fill_normal, line_dark))
    return handles


def add_top_order_legend(fig, ax_left, ax_right, orders, colors, light_colors, dark_colors):
    fig.canvas.draw()  # Must draw to get positions right before getting locations
    # Get the corner of the upper right plot in display coordinates
    upper_right_display = ax_right.transAxes.transform((1, 1))
    # Now put it in axes[0,0] coords
    upper_right_axes00 = ax_left.transAxes.inverted().transform(upper_right_display)
    handlers_ords = compute_filled_handles(colors, light_colors, dark_colors)
    # Must use axes[0,0] legend for constrained layout to work with it
    return ax_left.legend(
        handlers_ords, orders, ncol=4,
        loc='lower left', bbox_to_anchor=(0, 1.02, upper_right_axes00[0], 1.),
        mode='expand',
        columnspacing=0,
        handletextpad=0.5,
        fancybox=False, borderaxespad=0,
        handler_map={tuple: OrderBandsHandler()}
    )


def plot_empirical_saturation(ax=None, facecolor='lightgray', edgecolor='gray', alpha=0.4, zorder=9, **kwargs):
    if ax is None:
        ax = plt.gca()
    from matplotlib.patches import Rectangle
    # From Drischler 2018 arXiv:1710.08220
    n0 = 0.164
    n0_std = 0.007
    y0 = -15.86
    y0_std = np.sqrt(0.37 ** 2 + 0.2 ** 2)
    left = n0 - n0_std
    right = n0 + n0_std
    rect = Rectangle(
        (left, y0 - y0_std), width=right - left, height=2 * y0_std,
        facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder, **kwargs
    )
    ax.add_patch(rect)
    return ax
