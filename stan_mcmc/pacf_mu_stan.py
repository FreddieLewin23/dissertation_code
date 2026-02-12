import numpy as np
from partial_autocorrelation import pacf_to_phi
import pandas as pd
import matplotlib.pyplot as plt
import stan
from pathlib import Path
from partial_autocorrelation import phi_to_pacf


def fit_pacf_ar_with_mean_stan(y, p,
                               alpha=2.0, beta=2.0,
                               mu0=0.0, c2=1e6,
                               a0=2.0, b0_ig=1.0,
                               seed=42,
                               num_chains=4,
                               num_samples=1000,
                               num_warmup=1000):

    stan_path = Path(__file__).with_name("mu_pacf_pystan.stan")
    stan_code = stan_path.read_text()

    y = np.asarray(y, float).ravel()

    data = {
        "n": len(y),
        "p": p,
        "y": y,
        "alpha": alpha,
        "beta": beta,
        "mu0": mu0,
        "c2": c2,
        "a0": a0,
        "b0_ig": b0_ig,
    }

    posterior = stan.build(
        stan_code,
        data=data,
        random_seed=seed
    )

    fit = posterior.sample(
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup
    )

    return fit


def stan_fit_to_res_dict(fit, p: int) -> dict:
    """
    Convert PyStan3 fit object into the dict expected by plot_gibbs_pacf_traces.
    Requires the Stan model to output parameters:
      - kappa (vector[p])
      - mu (real)
      - sigma2 (generated quantities) OR sigma (real, then we square it)
    """
    df = fit.to_frame()

    # kappa columns are named like kappa[1], ..., kappa[p]
    kappa_cols = [f"kappa.{i}" for i in range(1, p + 1)]
    missing = [c for c in kappa_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing kappa columns in Stan draws: {missing}. "
            "Make sure your Stan parameters include: vector[p] kappa;"
        )

    kappa_samps = df[kappa_cols].to_numpy()
    mu_samps = df["mu"].to_numpy() if "mu" in df.columns else None

    if "sigma2" in df.columns:
        sigma2_samps = df["sigma2"].to_numpy()
    elif "sigma" in df.columns:
        sigma2_samps = (df["sigma"].to_numpy()) ** 2
    else:
        raise KeyError(
            "Missing sigma2 (or sigma) in Stan draws. "
            "Make sure your Stan model has `real<lower=0> sigma;` "
            "and ideally `generated quantities { real sigma2 = square(sigma); }`."
        )

    return {
        "kappa_samples": kappa_samps,
        "mu_samples": mu_samps,
        "sigma2_samples": sigma2_samps,
        "accept_rate": np.nan,  # Stan doesn't report MH accept rate like your sampler
    }
