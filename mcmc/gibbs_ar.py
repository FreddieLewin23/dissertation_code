import pandas as pd
import numpy as np
from check_order_stationarity import _lagged_matrix
from dataclasses import dataclass
from typing import Optional, Tuple
from partial_autocorrelation import phi_to_pacf, pacf_to_phi
import yfinance as yf

def is_stationary_from_phi(phi: np.ndarray, eps: float = 1e-12) -> bool:
    p = len(phi)
    if p == 0:
        return True
    Phi = np.zeros((p, p), dtype=float)
    Phi[0, :] = phi
    if p > 1:
        Phi[1:, :-1] = np.eye(p - 1)
    eigs = np.linalg.eigvals(Phi)
    return np.all(np.abs(eigs) < 1.0 - eps)


@dataclass
class GibbsConfig:
    p: int
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 1
    add_intercept: bool = False       # include an intercept in AR model
    b0: Optional[np.ndarray] = None   # prior mean for beta (shape d,)
    V0: Optional[np.ndarray] = None   # prior scale for beta (d x d), multiplies sigma^2
    a0: float = 2.0                   # prior shape for sigma^2 (weakly informative)
    b0_ig: float = 1.0                # prior scale for sigma^2
    enforce_stationarity: bool = True
    max_resample_phi: int = 200       # max attempts to draw stationary phi per iteration
    rng_seed: Optional[int] = 0


@dataclass
class GibbsResult:
    beta_samples: np.ndarray     # (n_save, d) ; d = p (+1 if intercept)
    sigma2_samples: np.ndarray   # (n_save,)
    y_used: np.ndarray
    accepted_stationary_rate: float
    config: GibbsConfig


def gibbs_ar(y: np.ndarray, cfg: GibbsConfig) -> GibbsResult:
    """
    Gibbs sampler for Gaussian AR(p) with Normal-Inverse-Gamma prior.
    Model: y_t = c + sum_{i=1}^p phi_i y_{t-i} + e_t,  e_t ~ N(0, sigma^2)
           beta = [c? , phi_1..phi_p]
           beta | sigma2 ~ N(b0, sigma2 * V0)
           sigma2 ~ InvGamma(a0, b0_ig)
    If cfg.add_intercept=False, beta is just phi
    """
    y = np.asarray(y, float).ravel()
    y = y[~np.isnan(y)]
    y_used = y.copy()

    y_p, X = _lagged_matrix(y_used, cfg.p)
    n, d = X.shape

    if cfg.b0 is None:
        b0 = np.zeros(d)
    else:
        b0 = np.asarray(cfg.b0, float).ravel()
        assert b0.shape == (d,)

    if cfg.V0 is None:
        tau2 = 100.0
        V0 = np.eye(d) * tau2
    else:
        V0 = np.asarray(cfg.V0, float)
        assert V0.shape == (d, d)

    a0 = float(cfg.a0)
    b0_ig = float(cfg.b0_ig)

    XtX = X.T @ X
    Xty = X.T @ y_p
    V0_inv = np.linalg.inv(V0)
    rng = np.random.default_rng(cfg.rng_seed)

    if d > 0:
        beta = np.zeros(d)  # start with phii = 0 (white noise)
    else:
        beta = np.zeros(0)  # no parameters for AR(0)

    sigma2 = float(np.var(y_p)) if np.var(y_p) > 0 else 1.0
    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    beta_samps = np.empty((n_save, d))
    s2_samps = np.empty(n_save)

    save_i = 0
    stationary_accepts = 0
    total_phi_draws = 0

    for iters in range(cfg.n_iter):
        # conditional for beta | sigma2, y: Normal(mn, sigma2 * Vn)
        Vn_inv = XtX + V0_inv
        Vn = np.linalg.inv(Vn_inv)
        mn = Vn @ (Xty + V0_inv @ b0)

        draw_attempts = 0
        while True:
            z = rng.standard_normal(d)
            beta_candidate = mn + np.linalg.cholesky(Vn) @ (np.sqrt(sigma2) * z)
            ok = True
            if cfg.enforce_stationarity:
                if cfg.add_intercept:
                    phi = beta_candidate[1:]  # drop intercept
                else:
                    phi = beta_candidate
                ok = is_stationary_from_phi(phi)
                draw_attempts += 1
                total_phi_draws += 1
                if draw_attempts > cfg.max_resample_phi:
                    # give up this iteration's beta draw; accept non-stationary draw to keep chain moving
                    ok = True
            if ok:
                beta = beta_candidate
                if cfg.enforce_stationarity and draw_attempts > 0:
                    stationary_accepts += 1
                break

        # conditional for sigma2 | beta, y: InvGamma(a_n, b_n)
        resid = y_p - X @ beta
        an = a0 + 0.5 * n
        bn = b0_ig + 0.5 * float(resid @ resid)

        sigma2 = 1.0 / rng.gamma(an, 1.0 / bn) # this is inverse gamma

        if iters >= cfg.burn and ((iters - cfg.burn) % cfg.thin == 0):
            beta_samps[save_i] = beta
            s2_samps[save_i] = sigma2
            save_i += 1

    acc_rate = (stationary_accepts / max(total_phi_draws, 1)) if cfg.enforce_stationarity else 1.0

    return GibbsResult(
        beta_samples=beta_samps,
        sigma2_samples=s2_samps,
        y_used=y_used,
        accepted_stationary_rate=acc_rate,
        config=cfg,
    )
