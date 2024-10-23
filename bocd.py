"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

This code is associated with the following blog posts:

    http://gregorygundersen.com/blog/2019/08/13/bocd/
    http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
============================================================================"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.stats import norm
from   scipy.special import logsumexp


# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T           = len(data)
    log_R       = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0              # log 0 == 1
    pmean       = np.empty(T)    # Model's predictive mean.
    pvar        = np.empty(T)    # Model's predictive variance.
    log_message = np.array([0])  # log 0 == 1
    log_H       = np.log(hazard)
    log_1mH     = np.log(1 - hazard)

    for t in range(1, T+1):
        # 2. Observe new datum.
        x = data[t-1]

        # Make model predictions.
        pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mean_params[:t])
        pvar[t-1]  = np.sum(np.exp(log_R[t-1, :t]) * model.var_params[:t])

        # 3. Evaluate predictive probabilities.
        # log_pis是个向量，计算当前点x 在各个 run length 下的概率密度
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        # 增长概率，计算当前点 x 在各个 run length 下的增长概率, log_growth_probs也是个向量
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        # 变点概率，计算当前点是变点的概率，是个标量
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        # 结合 4 和 5, 得出新的各个长度下的增长概率
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        # 归一化，让这些概率的和为1
        log_R[t, :t+1]  = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

        # print(log_message)

    R = np.exp(log_R)
    return R, pmean, pvar


# -----------------------------------------------------------------------------


class GaussianUnknownMean:

    def __init__(self, mean0, var0, varx):
        """Initialize model.

        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])

    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds  = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params  = self.prec_params + (1/self.varx)
        self.prec_params = np.append([1/self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params[:-1] +  (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx

# -----------------------------------------------------------------------------

def generate_data(varx, mean0, var0, T, cp_prob):
    """Generate partitioned data of T observations according to constant
    changepoint probability `cp_prob` with hyperpriors `mean0` and `prec0`.
    """
    data  = []
    cps   = []
    meanx = mean0
    for t in range(0, T):
        if np.random.random() < cp_prob:
            meanx = np.random.normal(mean0, var0)
            cps.append(t)
        data.append(np.random.normal(meanx, varx))
    return data, cps


# -----------------------------------------------------------------------------

def plot_posterior(T, data, cps, R, pmean, pvar):
    fig, axes = plt.subplots(2, 1, figsize=(20,10))

    ax1, ax2 = axes

    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)

    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', norm=LogNorm(vmin=0.001, vmax=1))
    ax2.set_xlim([0, T])
    # ax2.set_ylim([T, 0])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted')
        ax2.axvline(cp, c='red', ls='dotted')

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------


data2 = [259,179,181,205,184,182,168,192,184,175,152,178,156,183,162,180,168,191,194,209,201,183,197,199,168,165,170,187,178,179,195,186,171,176,191,168,172,170,170,191,199,192,201,185,176,194,184,172,188,201,194,162,196,209,211,191,189,179,171,221,277,277,228,234,227,230,243,278,264,267,258,281,302,278,242,269,279,254,250,249,258,260,264,276,294,284,311,237,276,285,253,278,276,283,255,258,267,265,277,268,254,263,274,272,231,232,259,244,250,242,226,244,235,228,243,223,218,261,218,292,312,299,238,132,79,73,68,73,68,73,85,87,72,64,74,76,66,57,69,67,66,77,61,76,228,360,279,230,246,247,226,223,245,218,237,206,226,204,232,225,247,211,209,208,212,245,220,187,186,208,178,192,190,198,209,180,176,196,190,293,328,325,900,800,367,323,285,265,241,243,262,257,249,206,240,238,232,262,224,221,219,221,222,212,216,212,251,214,241,226,234,248,232,226,246,226,220,240,223,234,216,234,215,263,254,226,243,218,207,209,256,221,238,231,237,222,234,240,250,263,285,253,238,228,220,215,208,196,231,235,207,205,186,217,223,227,179,210,199,215,195,221,194,185,212,217,190,200,205,207,177,197,204,197,176,190,201,195,206,214,159,195,207,198,226,216,192,196,187,194,180,184,192,202,213,155,178,212,176,223,240]
mean2 = 178


if __name__ == '__main__':
    data = data2
    mean0 = mean2

    T      = len(data)   # Number of observations.
    hazard = 1/1000  # Constant prior on changepoint probability.
    # mean0  = 744      # The prior mean on the mean parameter.
    var0   = 861      # The prior variance for mean parameter.
    varx   = 861      # The known variance of the data.

    # data, cps      = generate_data(varx, mean0, var0, T, hazard)
    model          = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(data, model, hazard)

    print(pvar)


    plot_posterior(T, data, [], R, pmean, pvar)
