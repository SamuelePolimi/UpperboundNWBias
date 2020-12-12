import numpy as np
import matplotlib as mp
import warnings

warnings.simplefilter("ignore")

mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{amsfonts}"]
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


from core.kernels import GaussianKernel, Triangle, Box
from core.functions import Laplace, Normal, Sin, Root, LnCosh, Cauchy, Uniform, Pareto, Log
from core.experiment import Experiment
from core.plot import PlotResult, PlotResultBias

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

print("imports")

plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.rc('lines', linewidth=0.5)


def experiment_1():
    title = "(c) Gaussian $h = 0.3$"
    y_label = r"$m(x)\!=\!\frac{\log\!\cosh(60x)}{60}$"
    y_design = r"$\mathrm{Laplace}(0, 0.5)$"
    x_grid = np.linspace(-1.5, 1.5, 500)
    design = Laplace(0., 0.5)
    regression_function = LnCosh(a=60., c=-1.)
    kernel = GaussianKernel(np.array([0.3]))
    print("before first exp")
    experiment = Experiment(regression_function, design, kernel)
    print("after first exp")

    zoom = dict(x_lim=(0.97, 1.03), y_lim=(0.16, 0.22), factor=8., loc='upper left', loc_1=1, loc_2=4)
    return experiment(x_grid), title, [-1.5, 1.5], zoom, y_label, y_design


def experiment_2():
    title = "(b) Gaussian $h = 0.2$"
    y_label = r"$m(x) = \log(x)$"
    y_design = r"$\mathrm{Pareto}(1, 1)$"
    x_grid = np.linspace(0.5, 3., 200)
    design = Pareto()
    regression_function = Log()
    kernel = GaussianKernel(np.array([0.2]))
    experiment = Experiment(regression_function, design, kernel)

    zoom = dict(x_lim=(0.97, 1.08), y_lim=(0.03, 0.11), factor=5, loc='upper right', loc_1=2, loc_2=3)
    return experiment(x_grid), title, (0.5, 3.), zoom, y_label, y_design


def experiment_3():
    title = "(a) Gaussian $h = 0.1$"
    y_label = r"$m(x) = \sqrt{x^2 + 1}$"
    y_design = r"$\mathrm{Cauchy}(0, 1)$"
    x_grid = np.linspace(-np.pi, np.pi, 200)
    design = Cauchy()
    regression_function = Root(scale=1.)
    kernel = GaussianKernel(np.array([0.1]))
    experiment = Experiment(regression_function, design, kernel)

    return experiment(x_grid), title, None, None, y_label, y_design


def experiment_4():
    title = "(d) Triangle $h = 0.3$"
    y_label = "$m(x) = \sin(5x)$"
    y_design = r"$\mathrm{Pareto}(1)$"
    x_grid = np.linspace(0., np.pi, 200)
    sigma = 0.2
    regression_function = Sin(frequency=5., amplitude=1., phase=0.)
    design = Pareto()#Normal(0., sigma)
    delta_function = lambda X: np.abs(X) + sigma
    kernel = Triangle(np.array([0.3]))
    experiment = Experiment(regression_function, design, kernel,
                            delta_function=delta_function)

    zoom = None#dict(x_lim=(0.95, 1.2), y_lim=(-1.4, 0), factor=2.5, loc=4, loc_1=2, loc_2=3)
    return experiment(x_grid), title, (0., np.pi), zoom, y_label, y_design


def experiment_5():
    title = "(e) Box $h = 0.4$"
    a, b = -2., 2.
    y_label = "$m(x) = \sin(5x)$"
    y_design = r"$\mathrm{Uniform}(-2, 2)$"
    x_grid = np.linspace(-np.pi, np.pi, 200)
    design = Uniform(a, b)
    regression_function = Sin(frequency=5., amplitude=1., phase=0.)
    x_grid = np.linspace(-np.pi, np.pi, 200)

    kernel = Box(np.array([0.4]))
    experiment = Experiment(regression_function, design, kernel)
    return experiment(x_grid), title, None, None, y_label, y_design


experiments = [
    experiment_3,
    experiment_2,
    experiment_1,
    experiment_4,
    experiment_5
]

fig = plt.figure(figsize=(6, 7))
grid = plt.GridSpec(27, 2) # plt.GridSpec(9, 2)
#, hspace=0.2, wspace=0.2)
#fig.suptitle(r"$f_x= Laplace(\lambda=0.5), L_m=%.2f, L_f=%.2f$" % (regression_function.global_lipschitz(),
#                                                                   design.log_lipschitz()))
plots = [
    (0, 0),
    (0, 1),
    (9, 0),
    (9, 1),
    (18, 0)]

for e, p in zip(experiments, plots):
    print("exp1")
    result, title, x_lim, zoom, y_label, y_design = e()

    ax = fig.add_subplot(grid[p[0]+1:p[0]+4, p[1]], xticklabels=[])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(y_label, fontsize=4.5)

    ax.set_xticks([])
    legend = PlotResult(ax, result, title, show_rosenblatt=False, show_tosatto=False, show_desing=False, x_lim=x_lim, color=True)

    ax = fig.add_subplot(grid[p[0] + 4:p[0] + 7, p[1]], xticklabels=[])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(r"$\mathrm{Bias}$", fontsize=4.5)

    legend = PlotResultBias(ax, result, title, show_desing=False, x_lim=x_lim, ret=legend, color=True)
    if zoom is not None:
        axins = zoomed_inset_axes(ax, zoom["factor"], loc=zoom["loc"])
        PlotResultBias(axins, result, title, show_desing=False, x_lim=zoom["x_lim"], ret=legend, color=True)
        axins.set_xlim(*zoom["x_lim"])
        axins.set_ylim(*zoom["y_lim"])
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax, axins, loc1=zoom["loc_1"], loc2=zoom["loc_2"], fc="none", ec="0.5")

    ax = fig.add_subplot(grid[p[0]+7:p[0]+9, p[1]])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(y_design, fontsize=4.5)
    PlotResult(ax, result, "",
               show_rosenblatt=False,
               show_tosatto=False,
               show_prediction=False,
               show_desing=True, x_lim=x_lim, ret=legend, color=True)

# for ax in axs.flat:
#     ax.set(xlabel='x', ylabel='y')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

ax = fig.add_subplot(grid[19:22, 1], xticklabels=[])
leg = ax.legend(list(legend.values())[:2], list(legend.keys())[:2], fontsize=6, loc="upper center",
          bbox_to_anchor=(0.1, 0, 0.7, 1.), mode="expand", title="Regression")

leg.set_title("Regression", prop={'size':6})
ax.axis('off')


ax = fig.add_subplot(grid[22:25, 1], xticklabels=[])
leg = ax.legend(list(legend.values())[2:5], list(legend.keys())[2:5], fontsize=6, loc="upper center",
          bbox_to_anchor=(0.1, 0.1, 0.7, 1.), mode="expand", title="Bias")
leg.set_title("Bias", prop={'size':6})
ax.axis('off')


ax = fig.add_subplot(grid[25:, 1], xticklabels=[])
leg = ax.legend(list(legend.values())[5:], list(legend.keys())[5:], fontsize=6, loc="upper center",
          bbox_to_anchor=(0.1, 0, 0.7, 1.), mode="expand", title="Design")
leg.set_title("Design", prop={'size':6})
ax.axis('off')

plt.savefig("plots/multi.pdf", bbox_inches='tight',
    pad_inches=0.05)
plt.show()
