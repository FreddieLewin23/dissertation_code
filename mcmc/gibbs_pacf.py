from check_order_stationarity import _lagged_matrix
from partial_autocorrelation import pacf_to_phi
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional
from dataclasses import dataclass


@dataclass
class GibbsPACFConfig:
    p: int
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 1
    alpha_beta_prior: tuple = (2.0, 2.0)
    prop_sd: float = 0.01
    a0: float = 2.0
    b0_ig: float = 1.0
    rng_seed: Optional[int] = 42


def kappa_to_z(kappa: np.ndarray) -> np.ndarray:
    return kappa / np.sqrt(1.0 - kappa**2)


def z_to_kappa(z: np.ndarray) -> np.ndarray:
    return z / np.sqrt(1.0 + z**2)


def gibbs_ar_pacf(y: np.ndarray, cfg: GibbsPACFConfig):
    rng = np.random.default_rng(cfg.rng_seed)

    y = np.asarray(y, float).ravel()
    y_p, X = _lagged_matrix(y, cfg.p)
    n, p = X.shape

    # initialise
    kappa = np.zeros(p)
    z = kappa_to_z(kappa)
    phi = pacf_to_phi(kappa)
    sigma2 = np.var(y_p)

    alpha, beta = cfg.alpha_beta_prior

    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.empty((n_save, p))
    sigma2_samps = np.empty(n_save)

    accept_count = 0
    total_proposals = 0
    save_i = 0

    def log_prior_kappa(k):
        x = 0.5 * (k + 1.0)  # map (-1,1) -> (0,1)
        return np.sum(beta_dist.logpdf(x, alpha, beta))

    def cond_loglik(phi_vec, s2):
        r = y_p - X @ phi_vec
        return -0.5 * (r @ r) / s2

    for it in range(cfg.n_iter):

        # ---------- Step 1: MH update for kappa ----------
        total_proposals += 1

        ll_curr = cond_loglik(phi, sigma2)
        lp_curr = log_prior_kappa(kappa)

        # propose in z-space

        z_prop = z + rng.normal(0.0, cfg.prop_sd, size=p)
        kappa_prop = z_to_kappa(z_prop)
        phi_prop = pacf_to_phi(kappa_prop)

        ll_prop = cond_loglik(phi_prop, sigma2)
        lp_prop = log_prior_kappa(kappa_prop)

        # Jacobian terms for z = kappa / sqrt(1 - kappa^2)
        log_jac_curr = -1.5 * np.sum(np.log(1.0 - kappa**2))
        log_jac_prop = -1.5 * np.sum(np.log(1.0 - kappa_prop**2))

        # IMPORTANT: correct direction for proposal ratio
        log_q_ratio = log_jac_prop - log_jac_curr

        log_acc_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr) + log_q_ratio

        if np.log(rng.uniform()) < log_acc_ratio:
            kappa = kappa_prop
            z = z_prop
            phi = phi_prop
            accept_count += 1



        # ---------- Step 2: Gibbs update for sigma^2 ----------
        resid = y_p - X @ phi
        an = cfg.a0 + 0.5 * n
        bn = cfg.b0_ig + 0.5 * (resid @ resid)
        sigma2 = 1.0 / rng.gamma(an, 1.0 / bn)

        # ---------- save ----------
        if it >= cfg.burn and ((it - cfg.burn) % cfg.thin == 0):
            kappa_samps[save_i] = kappa
            sigma2_samps[save_i] = sigma2
            save_i += 1

    accept_rate = accept_count / max(total_proposals, 1)

    return {
        "kappa_samples": kappa_samps,
        "sigma2_samples": sigma2_samps,
        "accept_rate": accept_rate,
    }
