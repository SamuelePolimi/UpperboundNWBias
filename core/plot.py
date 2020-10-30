import numpy as np
from matplotlib.pyplot import Axes

from .experiment import ExperimentResult


def PlotResult(ax, result, title="", show_tosatto=True, show_rosenblatt=True, show_desing=True, show_prediction=True,
               show_data_lims=False, color=True, x_lim=None, y_lim=None, line_width=0.75, ret={}):
    """
    Th
    :param ax:
    :type ax: Axes
    :param result:
    :type result: ExperimentResult
    :param title:
    :param show_desing:
    :param show_data_lims:
    :return:
    """
    colors = {
        'tosatto': (0.3, 0.3, 1.) if color else 'k',
        'rosenblatt': (0.3, 1., 0.3) if color else 'k',
        'true': (1., 0.3, 0.3) if color else 'k',
        'prediction': 'black' if color else 'k'
    }

    if show_tosatto:
        ret["Tosatto et al."] = ax.fill_between(result.x_grid,
                         result.y_pred - result.b_tosatto,
                         result.y_pred + result.b_tosatto,
                                facecolor=colors['tosatto'],
                                ls="--",
 #                               hatch="...",
                                edgecolor=colors['tosatto'],
                                alpha=0.75)
    if show_rosenblatt:
        ret["Rosenblatt"] = ax.fill_between(result.x_grid,
                         result.y_pred - result.b_rosenblatt,
                         result.y_pred + result.b_rosenblatt,
                                facecolor=colors['rosenblatt'],
                                ls="--",
#                                hatch=r"////",
                                edgecolor=colors['rosenblatt'],
                                alpha=0.75)

    if show_prediction:
        ret["$\mathbb{E}[\hat{m}(x)]$"] = ax.plot(result.x_grid, result.y_pred, ls="-.", color=colors['prediction'],
                                                  linewidth=line_width)[0]
        ret["$m(x)$"] = ax.plot(result.x_grid, result.y_true, color=colors['true'], linewidth=line_width)[0]

    if show_desing:
        ret["$f_X(x)$"] = ax.plot(result.x_grid, result.y_design, ls=':', color="k")[0]

    if show_data_lims:
        x_min = result.x_min
        x_max = result.x_max

        ax.axvline(x_min)
        ax.axvline(x_max)

    #ax.set_xlim(1., 10.)
    if x_lim is None:
        ax.set_xlim(-np.pi, np.pi)
    else:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_title(title, fontsize=6)

    return ret


def PlotResultBias(ax, result, title="", show_tosatto=True, show_rosenblatt=True, show_desing=True, show_prediction=True,
               show_data_lims=False, color=True, x_lim=None, y_lim=None, ret={}):
    """
    Th
    :param ax:
    :type ax: Axes
    :param result:
    :type result: ExperimentResult
    :param title:
    :param show_desing:
    :param show_data_lims:
    :return:
    """
    colors = {
        'tosatto': (0.3, 0.3, 1.) if color else 'k',
        'rosenblatt': (0.3, 1., 0.3) if color else 'k',
        'true': (0., 0., 0.) if color else 'k',
        'prediction': 'black' if color else 'k'
    }

    if show_tosatto:
        ret["Tosatto et al."] = ax.plot(result.x_grid, np.abs(result.b_tosatto),
                                #facecolor=colors['tosatto'],
                                ls="-.",
                                color=colors['tosatto'])[0]
    if show_rosenblatt:
        ret["Rosenblatt"] = ax.plot(result.x_grid, np.abs(result.b_rosenblatt),
                                #facecolor=colors['rosenblatt'],
                                ls="--",
                                color=colors['rosenblatt'])[0]

    if show_prediction:
        ret[r"$|\mathbb{E}[\hat{m}(x)] - m(x)|$"] = ax.plot(result.x_grid, np.abs(result.y_true - result.y_pred), #ls="-.",
                                                  color=colors['prediction'],
                                                  linewidth=0.5)[0]
        #ret["$m(x)$"] = ax.plot(result.x_grid, result.y_true, color=colors['true'])[0]

    if show_desing:
        ret["$f_X(x)$"] = ax.plot(result.x_grid, result.y_design, ls=':', color="k")[0]

    if show_data_lims:
        x_min = result.x_min
        x_max = result.x_max

        ax.axvline(x_min)
        ax.axvline(x_max)

    #ax.set_xlim(1., 10.)
    if x_lim is None:
        ax.set_xlim(-np.pi, np.pi)
    else:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    #ax.set_title(title, fontsize=6)

    return ret