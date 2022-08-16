#%%

import arviz as az
import numpy as np
import numpyro
from iminuit import Minuit
from jax.random import PRNGKey as Key
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import beta as sp_beta
from scipy.stats import betabinom as sp_betabinom
from scipy.stats import expon as sp_exponential


#%%
#
numpyro.enable_x64()

SEED = 42

k = np.array([2036195, 745632, 279947, 200865, 106383, 150621])
N = np.array([7642688, 7609177, 8992872, 8679915, 8877887, 8669401])
x = np.arange(1, len(N) + 1)


#%%

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, k/N, color="C3")
ax.set(xlabel="x", ylabel="k / N", title="Post Mortem Damage")
fig.savefig("postmortem_damage.pdf", transparent=True)

#%%


def mu_phi_to_alpha_beta(mu, phi):
    alpha = mu * phi
    beta = phi * (1 - mu)
    return alpha, beta


# mean, concentration
A_prior = mu_phi_to_alpha_beta(mu=0.2, phi=5)
q_prior = mu_phi_to_alpha_beta(mu=0.2, phi=5)
c_prior = mu_phi_to_alpha_beta(mu=0.1, phi=10)
phi_prior = (2, 1000)  # (min, scale)


#%%


def model(x, N, k=None):

    A = numpyro.sample("A", dist.Beta(*A_prior))
    q = numpyro.sample("q", dist.Beta(*q_prior))
    c = numpyro.sample("c", dist.Beta(*c_prior))
    D_x = numpyro.deterministic("Dx", A * (1 - q) ** (x - 1) + c)

    delta = numpyro.sample("delta", dist.Exponential(1 / phi_prior[1]))
    phi = numpyro.deterministic("phi", delta + phi_prior[0])

    alpha = numpyro.deterministic("alpha", D_x * phi)
    beta = numpyro.deterministic("beta", (1 - D_x) * phi)

    numpyro.sample("obs", dist.BetaBinomial(alpha, beta, N), obs=k)


def get_samples():

    mcmc = MCMC(
        NUTS(model),
        num_warmup=1000,
        num_samples=1000,
        progress_bar=False,
    )
    mcmc.run(Key(SEED), x, N, k)

    samples = mcmc.get_samples()

    return mcmc, samples


mcmc, samples = get_samples()
mcmc.print_summary()

# %timeit get_samples()
# 2.12 s ± 13.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

#%%

data = az.from_numpyro(mcmc)
az.plot_trace(
    data,
    figsize=(10, 16),
    var_names=("A", "q", "c", "phi"),
)

# %%

# %%

# A = 0.25
# q = 0.7
# c = 0.001
# phi = 800


def log_likelihood_PMD(A, q, c, phi, x, k, N):
    D_x = A * (1 - q) ** (x - 1) + c
    alpha = D_x * phi
    beta = (1 - D_x) * phi
    return -sp_betabinom.logpmf(k, N, alpha, beta).sum()


def log_prior_PMD(A, q, c, phi):
    lp = (
        sp_beta.logpdf(A, *A_prior)
        + sp_beta.logpdf(q, *q_prior)
        + sp_beta.logpdf(c, *c_prior)
        + sp_exponential.logpdf(phi, *phi_prior)
    )
    return -lp


def log_posterior_PMD(A, q, c, phi, x, k, N):
    log_likelihood = log_likelihood_PMD(A, q, c, phi, x, k, N)
    log_p = log_prior_PMD(A, q, c, phi)
    return log_likelihood + log_p


def cost_function_MLE(A, q, c, phi):
    return log_likelihood_PMD(A, q, c, phi, x, k, N)


def cost_function_MAP(A, q, c, phi):
    return log_posterior_PMD(A, q, c, phi, x, k, N)


#%%


def fit_iminuit(cost_function):
    m = Minuit(cost_function, A=0.1, q=0.5, c=0.01, phi=1000)
    m.limits["A"] = (0, 1)
    m.limits["q"] = (0, 1)
    m.limits["c"] = (0, 1)
    m.limits["phi"] = (2, None)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    return m


m_MLE = fit_iminuit(cost_function_MLE)
m_MAP = fit_iminuit(cost_function_MAP)


# %timeit fit_iminuit(cost_function_MLE)
# 9.55 ms ± 140 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# %timeit fit_iminuit(cost_function_MAP)
# 61.6 ms ± 630 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

#%%

from numba import njit


A = 0.25
q = 0.7
c = 0.001
phi = 800

cost_function_MAP_jit = njit(cost_function_MAP)
cost_function_MAP_jit(A, q, c, phi)


# %%


@njit
def log_likelihood_PMD_jit(A, q, c, phi, x, k, N):
    D_x = A * (1 - q) ** (x - 1) + c
    alpha = D_x * phi
    beta = (1 - D_x) * phi
    return -sp_betabinom.logpmf(k, N, alpha, beta).sum()


log_likelihood_PMD_jit(A, q, c, phi, x, k, N)

#%%

from scipy.special import betaln


@njit
def betabinom_logpmf(x, n, a, b):
    k = np.floor(x)
    combiln = -np.log(n + 1) - betaln(n - k + 1, k + 1)
    return combiln + betaln(k + a, n - k + b) - betaln(a, b)


@njit
def log_likelihood_PMD_jit(A, q, c, phi, x, k, N):
    D_x = A * (1 - q) ** (x - 1) + c
    alpha = D_x * phi
    beta = (1 - D_x) * phi
    return -betabinom_logpmf(k, N, alpha, beta).sum()


log_likelihood_PMD_jit(A, q, c, phi, x, k, N)


#%%

from numba_scipy.special import betaln


@njit
def betabinom_logpmf(x, n, a, b):
    k = np.floor(x)
    combiln = -np.log(n + 1) - betaln(n - k + 1, k + 1)
    return combiln + betaln(k + a, n - k + b) - betaln(a, b)


@njit
def log_likelihood_PMD_jit(A, q, c, phi, x, k, N):
    D_x = A * (1 - q) ** (x - 1) + c
    alpha = D_x * phi
    beta = (1 - D_x) * phi
    return -betabinom_logpmf(k, N, alpha, beta).sum()


log_likelihood_PMD_jit(A, q, c, phi, x, k, N)


#%%

from jax.scipy.special import betaln


@njit
def betabinom_logpmf(x, n, a, b):
    k = np.floor(x)
    combiln = -np.log(n + 1) - betaln(n - k + 1, k + 1)
    return combiln + betaln(k + a, n - k + b) - betaln(a, b)


@njit
def log_likelihood_PMD_jit(A, q, c, phi, x, k, N):
    D_x = A * (1 - q) ** (x - 1) + c
    alpha = D_x * phi
    beta = (1 - D_x) * phi
    return -betabinom_logpmf(k, N, alpha, beta).sum()


log_likelihood_PMD_jit(A, q, c, phi, x, k, N)


#%%

# https://en.wikipedia.org/wiki/Beta-binomial_distribution#Motivation_and_derivation


import math


@njit
def gammaln_vec_jit(xs):
    out = np.empty(len(xs), dtype="float")
    for i, x in enumerate(xs):
        out[i] = math.lgamma(x)
    return out


@njit
def log_betabinom_jit(k, N, alpha, beta):
    return (
        gammaln_vec_jit(N + 1)  # type: ignore
        + gammaln_vec_jit(k + alpha)
        + gammaln_vec_jit(N - k + beta)
        + gammaln_vec_jit(alpha + beta)
        - (
            gammaln_vec_jit(k + 1)
            + gammaln_vec_jit(N - k + 1)
            + gammaln_vec_jit(alpha)
            + gammaln_vec_jit(beta)
            + gammaln_vec_jit(N + alpha + beta)
        )
    )


@njit
def log_likelihood_PMD_jit(A, q, c, phi, x, k, N):
    D_x = A * (1 - q) ** (x - 1) + c
    alpha = D_x * phi
    beta = (1 - D_x) * phi
    return -log_betabinom_jit(k, N, alpha, beta).sum()


log_likelihood_PMD_jit(A, q, c, phi, x, k, N)


#%%


@njit
def xlog1py_jit(x, y):
    if x == 0:
        return 0

    return x * np.log1p(y)


@njit
def xlogy_jit(x, y):
    if x == 0:
        return 0

    return x * np.log(y)


@njit
def gammaln_scalar_jit(x):
    return math.lgamma(x)


@njit
def betaln_jit(x, y):
    return gammaln_scalar_jit(x) + gammaln_scalar_jit(y) - gammaln_scalar_jit(x + y)


@njit
def log_beta_jit(x, alpha, beta):
    lPx = xlog1py_jit(beta - 1.0, -x) + xlogy_jit(alpha - 1.0, x)
    lPx -= betaln_jit(alpha, beta)
    return lPx


@njit
def log_exponential_jit(x, loc, scale):
    if x < loc:
        return -np.inf
    return -(x - loc) / scale - np.log(scale)


@njit
def log_prior_PMD_jit(A, q, c, phi):
    lp = (
        log_beta_jit(A, *A_prior)
        + log_beta_jit(q, *q_prior)
        + log_beta_jit(c, *c_prior)
        + log_exponential_jit(phi, *phi_prior)
    )
    return -lp


log_prior_PMD_jit(A, q, c, phi)

# %%


@njit
def log_posterior_PMD_jit(A, q, c, phi, x, k, N):
    log_likelihood = log_likelihood_PMD_jit(A, q, c, phi, x, k, N)
    log_p = log_prior_PMD_jit(A, q, c, phi)
    return log_likelihood + log_p


log_posterior_PMD_jit(A, q, c, phi, x, k, N)


@njit
def cost_function_MLE_jit(A, q, c, phi):
    return log_likelihood_PMD_jit(A, q, c, phi, x, k, N)


cost_function_MLE_jit(A, q, c, phi)


@njit
def cost_function_MAP_jit(A, q, c, phi):
    return log_posterior_PMD_jit(A, q, c, phi, x, k, N)


cost_function_MAP_jit(A, q, c, phi)

# %%


m_MLE_jit = fit_iminuit(cost_function_MLE_jit)
m_MAP_jit = fit_iminuit(cost_function_MAP_jit)


# %timeit fit_iminuit(cost_function_MLE_jit)
# 1.19 ms ± 13.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
# %timeit fit_iminuit(cost_function_MAP_jit)
# 1.21 ms ± 8.8 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# %%
