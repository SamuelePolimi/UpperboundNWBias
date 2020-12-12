import numpy as np
import matplotlib as mp

mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{amsfonts}"]
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)

from core.bias_estimation import BiasEstimator, BiasTosatto
from core.kernels import GaussianKernel
from core.kernel_regression import KernelRegression
from core.ensemble_estimator import Ensemble
from core.functions import Laplace, Normal, Sin, Lin, Root, LnCosh, Cauchy, Uniform, Pareto, Log, MultiUniform, MultiGaussian
from core.sampler import Sampler

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('lines', linewidth=2)

colors = {
    'tosatto': (0.3, 0.3, 1.),
    'rosenblatt': (0.3, 1., 0.3),
    'true': (0., 0., 0.),
    'prediction': 'black'
}

n_dims = range(1, 10)

fix, axs = plt.subplots(1, 3, figsize=(9, 3))
#TODO: change back to 0.
for v_query, ax in zip([0, 0.5, 0.75], axs):

    list_tosatto_bias = []
    list_true_bias = []
    for d in n_dims:
        ker = GaussianKernel(0.1*np.ones(d))

        lin = Lin(d)

        X = np.linspace(np.zeros(d), np.ones(d))
        y = lin(X)
        uniform = MultiGaussian(np.zeros(d), np.ones(d)) # MultiUniform(-np.ones(d), np.ones(d)) #
        # sin = Sin(1., 1., 0.)
        # y = sin(X)
        ker_reg = KernelRegression(ker)
        ker_reg.fit(X, y)
        # print(y)
        query = np.ones(d)*v_query

        #
        # ens = Ensemble(lambda: KernelRegression(ker), Sampler(uniform, lin, None),
        #                              n_samples=1000, n_models=10)
        # ens = Ensemble(lambda: KernelRegression(ker), Sampler(uniform, lin, None),
        #               n_samples=100000, n_models=1000)
        ens = Ensemble(lambda: KernelRegression(ker), Sampler(uniform, lin, None),
                       n_samples=100000, n_models=1000)
        true_bias = np.abs(lin(np.array([query])) - ens.predict(np.array([query])))
        list_true_bias.append(true_bias)
        print("true bias", true_bias)
        bias = BiasTosatto(ker, uniform, lin)
        tosatto_bias = bias(np.array([query]))
        print("bias tosatto", tosatto_bias)
        list_tosatto_bias.append(tosatto_bias)

    ax.set_title("$\mathbf{x}=%.2f$" % v_query)
    ax.plot(n_dims, list_tosatto_bias, label="Tosatto et al.", ls="-.", color=colors["tosatto"])
    ax.plot(n_dims, list_true_bias, label="$|\mathbb{E}[\hat{m}(x)] - m(x)|$", color=colors["true"])
    if v_query==0.:
        ax.set_ylabel("Bias")
    if v_query==0.75:
        ax.legend(loc="best")
    ax.set_xlabel("$d$")
plt.savefig("plots/figure2.pdf", bbox_inches='tight',
    pad_inches=0.05)
plt.show()