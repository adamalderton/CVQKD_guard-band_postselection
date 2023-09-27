from typing import Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mrmustard import settings
from mrmustard.physics.fock import quadrature_distribution
from mrmustard.utils.wigner import wigner_discretized

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def _wigner_and_rho_plot(
    rho: np.ndarray,
    xbounds: Tuple[int] = (-6, 6),
    ybounds: Tuple[int] = (-6, 6),
    **kwargs,
):  # pylint: disable=too-many-statements
    """
    *** ADAPTED FROM XANADU'S MRMUSTARD ***
    See: https://github.com/XanaduAI/MrMustard.
    
    Plots the Wigner function of a state given its density matrix.

    Args:
        rho (np.ndarray): density matrix of the state
        xbounds (Tuple[int]): range of the x axis
        ybounds (Tuple[int]): range of the y axis

    Keyword args:
        resolution (int): number of points used to calculate the wigner function
        xticks (Tuple[int]): ticks of the x axis
        xtick_labels (Optional[Tuple[str]]): labels of the x axis; if None uses default formatter
        yticks (Tuple[int]): ticks of the y axis
        ytick_labels (Optional[Tuple[str]]): labels of the y axis; if None uses default formatter
        grid (bool): whether to display the grid
        cmap (matplotlib.colormap): colormap of the figure

    Returns:
        tuple: figure and axes
    """

    # Need a diverging colourmap for things to look sensible. This is the (new) default.
    # Here I create my own with white at the centre and black at the extremities.
    colors = [(0, 0, 0), (1, 1, 1), (0, 0, 0)]  # Black, White, Black
    cmap_custom = matplotlib.mcolors.LinearSegmentedColormap.from_list('Custom', colors, N=256)

    plot_args = {
        "resolution": 200,
        "xticks": (-5, 0, 5),
        "xtick_labels": None,
        "yticks": (-5, 0, 5),
        "ytick_labels": None,
        "grid": False,
        "cmap": cmap_custom,
    }
    plot_args.update(kwargs)

    if plot_args["xtick_labels"] is None:
        plot_args["xtick_labels"] = plot_args["xticks"]
    if plot_args["ytick_labels"] is None:
        plot_args["ytick_labels"] = plot_args["yticks"]

    q, ProbX = quadrature_distribution(rho)
    p, ProbP = quadrature_distribution(rho, np.pi / 2)

    xvec = np.linspace(*xbounds, plot_args["resolution"])
    pvec = np.linspace(*ybounds, plot_args["resolution"])
    W, X, P = wigner_discretized(rho, xvec, pvec, settings.HBAR)

    ### PLOTTING ###

    fig, ax = plt.subplots(
        2, 2, figsize=(6, 6), gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 2]}
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # Wigner function
    ax[1][0].contourf(X, P, W, 120, cmap=plot_args["cmap"], vmin=-abs(W).max(), vmax=abs(W).max())
    ax[1][0].set_xlabel("$x$", fontsize=12)
    ax[1][0].set_ylabel("$p$", fontsize=12)
    ax[1][0].get_xaxis().set_ticks(plot_args["xticks"])
    ax[1][0].xaxis.set_ticklabels(plot_args["xtick_labels"])
    ax[1][0].get_yaxis().set_ticks(plot_args["yticks"])
    ax[1][0].yaxis.set_ticklabels(plot_args["ytick_labels"], rotation="vertical", va="center")
    ax[1][0].tick_params(direction="in")
    ax[1][0].set_xlim(xbounds)
    ax[1][0].set_ylim(ybounds)
    ax[1][0].grid(plot_args["grid"])

    # X quadrature probability distribution
    ax[0][0].fill(q, ProbX, color=plot_args["cmap"](0.5))
    ax[0][0].plot(q, ProbX, color=plot_args["cmap"](0.8))
    # ax[0][0].plot(q, ProbX, color="black")
    ax[0][0].get_xaxis().set_ticks(plot_args["xticks"])
    ax[0][0].xaxis.set_ticklabels([])
    ax[0][0].get_yaxis().set_ticks([])
    ax[0][0].tick_params(direction="in")
    ax[0][0].set_ylabel("Prob($x$)", fontsize=12)
    ax[0][0].set_xlim(xbounds)
    ax[0][0].set_ylim([0, 1.1 * max(ProbX)])
    ax[0][0].grid(plot_args["grid"])

    # P quadrature probability distribution
    ax[1][1].fill(ProbP, p, color=plot_args["cmap"](0.5))
    ax[1][1].plot(ProbP, p, color=plot_args["cmap"](0.8))
    # ax[1][1].plot(ProbP, p, color="black")
    ax[1][1].get_xaxis().set_ticks([])
    ax[1][1].get_yaxis().set_ticks(plot_args["yticks"])
    ax[1][1].yaxis.set_ticklabels([])
    ax[1][1].tick_params(direction="in")
    ax[1][1].set_xlabel("Prob($p$)", fontsize=12)
    ax[1][1].set_xlim([0, 1.1 * max(ProbP)])
    ax[1][1].set_ylim(ybounds)
    ax[1][1].grid(plot_args["grid"])

    # Density matrix
    ax[0][1].matshow(abs(rho), cmap=plot_args["cmap"], vmin=-abs(rho).max(), vmax=abs(rho).max())
    ax[0][1].set_title(r"abs($\rho$)", fontsize=12)
    ax[0][1].tick_params(direction="in")
    ax[0][1].get_xaxis().set_ticks([])
    ax[0][1].get_yaxis().set_ticks([])
    ax[0][1].set_aspect("auto")
    ax[0][1].set_ylabel(f"cutoff = {len(rho)}", fontsize=12)
    ax[0][1].yaxis.set_label_position("right")

    return fig, ax

def wigner_and_rho_plot_to_pgf(filename: str, rho: np.ndarray, **kwargs):
    """
    A convienient wrapper to output generate a .pgf plot (latex/tikz friendly) from 
    a density matrix. The plot shows the Wigner function, the x and p quadrature marginals
    and the density matrix itself.

    Arguments: 
        filename: str
            The name of the file to save the plot to. Should end in .pgf.
        
        rho: np.ndarray
            The density matrix to plot.
        
        **kwargs: dict
            Keyword arguments to pass to the _wigner_and_rho_plot function.
    
    Returns:
        None (saves the file to the specified location)
    """

    fig, ax = _wigner_and_rho_plot(rho, **kwargs)
    fig.savefig(filename, bbox_inches = 'tight')

