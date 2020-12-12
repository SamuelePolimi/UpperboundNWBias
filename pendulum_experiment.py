import numpy as np

import matplotlib as mp
import matplotlib.pyplot as plt
mp.rc('text', usetex=True)
mp.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{amsfonts}"]
plt.rc('text', usetex=True)


from core.functions import Pendulum, MultiUniform
from core.kernels import GaussianKernel
from core.bias_estimation import BiasTosatto
from core.kernel_regression import KernelRegression


def estimate_lipschitz(X, Y):
    n = X.shape[0]
    max_d = -np.infty
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            else:
                delta = np.sum(np.abs(X[i]-X[j]))
                delta_y = np.abs(Y[i]-Y[j])
                if delta==0.:
                    continue
                der = delta_y/delta
                if der > max_d:
                    max_d = der
    return max_d

colors = {
    'tosatto': (0.3, 0.3, 1.),
    'rosenblatt': (0.3, 1., 0.3),
    'true': (0., 0., 0.),
    'prediction': 'black'
}

f = Pendulum()
alphas = np.linspace(-np.pi, np.pi, 200)
query = np.concatenate([alphas.reshape(-1, 1), np.zeros((200, 2))], axis=1)
k = GaussianKernel(np.array([0.1, 0.1, 0.1]))

uniform = MultiUniform(-f.u, f.u)

predictions = []
for i in range(100):
    print("Experiment", i)
    # X = uniform(50000)
    X = uniform(50000)
    kernel_regression = KernelRegression(k)
    Y = f(X)
    kernel_regression.fit(X, Y)
    predictions.append(kernel_regression.predict(query).ravel())
y = np.ravel(f(query))
true_bias = np.abs(y - np.mean(predictions, axis=0))
print("True Bias", true_bias)
# print(estimate_lipschitz(X, Y))
kernel_regression = KernelRegression(k)
bias_tosatto = BiasTosatto(k, uniform, f, kernel_regression)
bias_t = bias_tosatto(query)
print("Bias", bias_t)
fig1, ax1 = plt.subplots(figsize=(3.5, 3))
ax1.plot(alphas, bias_t.ravel(), label="Tosatto et al.", c=colors['tosatto'], ls='-.')
ax1.plot(alphas, true_bias, label="$|\mathbb{E}[\hat{m}(x) - m(x)]|$", c=colors['true'])
ax1.legend(loc='best')
ax1.set_xticks([-np.pi, 0, np.pi])
ax1.set_xticklabels(["$-\pi$", "$0$", "$\pi$"])
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"Bias", rotation=90)
plt.savefig("plots/pendulum.pdf", bbox_inches='tight',
    pad_inches=0.05)
plt.show()
