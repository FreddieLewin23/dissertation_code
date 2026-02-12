from partial_autocorrelation import pacf_to_phi
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional
from dataclasses import dataclass
from matplotlib import pyplot as plt
from scipy import linalg

def _lagged_matrix(y: np.ndarray, p: int):
    """
    Create (y_p, X_p) for AR(p): y_t = sum_{i=1}^p phi_i y_{t-i} + e_t
    Returns y_p (n_eff,), X_p (n_eff, p)
    """
    if p == 0:
        return y.copy(), np.empty((y.shape[0], 0))
    T = len(y)
    y_p = y[p:].copy()
    X_p = np.column_stack([y[p - i:T - i] for i in range(1, p + 1)])
    return y_p, X_p


def kappa_to_z(kappa: np.ndarray) -> np.ndarray:
    return kappa / np.sqrt(1.0 - kappa**2)


def z_to_kappa(z: np.ndarray) -> np.ndarray:
    return z / np.sqrt(1.0 + z**2)


@dataclass
class GibbsPACFConfigMean:
    p: int
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 1
    alpha_beta_prior: tuple = (2.0, 2.0)
    prop_sd: float = 0.01
    a0: float = 2.0
    b0_ig: float = 1.0
    rng_seed: Optional[int] = 42

    # mean prior: mu - N(mu0, c2)
    mu0: float = 0.0
    c2: float = 1e6


def gibbs_ar_pacf_with_mean(y: np.ndarray, cfg: GibbsPACFConfigMean):
    """
    Correct version that uses the exact likelihood under stationarity:
      y_{p+1:n} | y_{1:p}, (kappa, mu, sigma2)  ~ N(X phi + c mu 1, sigma2 I)
      y_{1:p}   | (kappa, mu, sigma2)          ~ N_p(mu 1_p, V(kappa,sigma2))
    where V(kappa,sigma2) = sigma2 * P0(phi) and P0 solves the discrete Lyapunov equation.
    """
    rng = np.random.default_rng(cfg.rng_seed)
    y = np.asarray(y, float).ravel()

    # data splits
    y_p, X = _lagged_matrix(y, cfg.p)  # y_p = y[p:], X uses lags from y
    n_star, p = X.shape
    y_init = y[:p].copy()             # y_{1:p}

    # --- helpers
    alpha, beta = cfg.alpha_beta_prior
    ones_star = np.ones(n_star)
    ones_p = np.ones(p)

    def log_prior_kappa(k: np.ndarray) -> float:
        # independent scaled Beta on each kappa_j via x=(k+1)/2 in (0,1)
        x = 0.5 * (k + 1.0)
        # constant Jacobian from scaling (1/2)^p cancels in MH ratio, so omit
        return float(np.sum(beta_dist.logpdf(x, alpha, beta)))

    def c_from_phi(ph: np.ndarray) -> float:
        return float(1.0 - np.sum(ph))

    def companion_A(phi: np.ndarray) -> np.ndarray:
        # state s_t = [x_t, x_{t-1}, ..., x_{t-p+1}]', where x_t = y_t - mu
        # x_t = phi' [x_{t-1},...,x_{t-p}] + eps_t
        A = np.zeros((p, p), dtype=float)
        A[0, :] = phi
        if p > 1:
            A[1:, :-1] = np.eye(p - 1)
        return A

    def stationary_P0(phi: np.ndarray) -> np.ndarray:
        """
        Solve P = A P A' + Q with Q = e1 e1' (i.e., innovation variance = 1).
        Then for general sigma2: P(sigma2) = sigma2 * P0.
        """
        A = companion_A(phi)
        Q = np.zeros((p, p), dtype=float)
        Q[0, 0] = 1.0
        # SciPy solves A X A^T - X + Q = 0  -> X = A X A^T + Q
        P0 = linalg.solve_discrete_lyapunov(A, Q)
        return P0

    def quad_and_logdet_init(phi: np.ndarray, mu: float, sigma2: float):
        """
        For y_init ~ N_p(mu 1, V), V = sigma2 * P0(phi)
        Return:
          quad = (y_init - mu 1)' V^{-1} (y_init - mu 1)
          logdetV = log|V|
          also return P0^{-1} and logdetP0 to reuse in mu/sigma2 steps.
        """
        P0 = stationary_P0(phi)
        # numerically stable inverse + logdet via Cholesky if possible
        try:
            L = np.linalg.cholesky(P0)
            logdetP0 = 2.0 * np.sum(np.log(np.diag(L)))
            # solve P0^{-1} b via cho_solve
            P0_inv = linalg.cho_solve((L, True), np.eye(p))
        except np.linalg.LinAlgError:
            sign, logdetP0 = np.linalg.slogdet(P0)
            if sign <= 0:
                raise ValueError("P0 not SPD; phi may be non-stationary or numerical issues.")
            P0_inv = np.linalg.inv(P0)

        d = (y_init - mu * ones_p)
        quad = (d @ P0_inv @ d) / sigma2                # because V^{-1} = (1/sigma2) P0^{-1}
        logdetV = p * np.log(sigma2) + logdetP0
        return float(quad), float(logdetV), P0_inv, float(logdetP0)

    def loglike_conditional(phi: np.ndarray, mu: float, sigma2: float) -> float:
        c = c_from_phi(phi)
        resid = (y_p - X @ phi) - c * mu * ones_star
        return float(-0.5 * np.sum(resid ** 2) / sigma2)

    def loglike_initial(phi: np.ndarray, mu: float, sigma2: float) -> float:
        quad, logdetV, _, _ = quad_and_logdet_init(phi, mu, sigma2)
        return float(-0.5 * (quad + logdetV))

    def logpost_kappa(phi: np.ndarray, kappa: np.ndarray, mu: float, sigma2: float) -> float:
        # posterior in kappa up to constant, using exact likelihood
        return (
            log_prior_kappa(kappa)
            + loglike_conditional(phi, mu, sigma2)
            + loglike_initial(phi, mu, sigma2)
        )

    # --- initialise
    kappa = np.zeros(p)
    z = kappa_to_z(kappa)
    phi = pacf_to_phi(kappa)

    mu = float(np.mean(y))  # reasonable init
    sigma2 = float(np.var(y, ddof=1))

    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.empty((n_save, p))
    mu_samps = np.empty(n_save)
    sigma2_samps = np.empty(n_save)

    accept_count = 0
    total_proposals = 0
    save_i = 0

    for it in range(cfg.n_iter):

        # ---------- Step 1: MH for kappa (via z), conditional on (mu, sigma2)
        total_proposals += 1

        # target in z-space: log p(kappa(z) | ...) + log|dkappa/dz|
        # if kappa = tanh(z), then log|dkappa/dz| = sum log(1 - kappa^2)
        kappa_curr = kappa
        phi_curr = phi
        logt_curr = logpost_kappa(phi_curr, kappa_curr, mu, sigma2) + 1.5 * float(np.sum(np.log(1.0 - kappa_curr**2)))

        # propose in z-space
        z_prop = z + rng.normal(0.0, cfg.prop_sd, size=p)
        kappa_prop = z_to_kappa(z_prop)
        phi_prop = pacf_to_phi(kappa_prop)
        logt_prop = logpost_kappa(phi_prop, kappa_prop, mu, sigma2) + 1.5 * float(np.sum(np.log(1.0 - kappa_prop**2)))

        log_acc_ratio = logt_prop - logt_curr

        if np.log(rng.uniform()) < log_acc_ratio:
            kappa = kappa_prop
            z = z_prop
            phi = phi_prop
            accept_count += 1
        else:
            # keep current
            phi = phi_curr
            kappa = kappa_curr

        # ---------- Step 2: Gibbs for mu | (kappa, sigma2, y)  (exact)
        c_curr = c_from_phi(phi)

        # conditional part: r = y_{p+1:n} - X phi = c mu 1 + eps
        r = y_p - X @ phi
        S = float(np.sum(r))

        # initial block part: y_{1:p} ~ N_p(mu 1, sigma2 P0)
        quad_dummy, logdet_dummy, P0_inv, _ = quad_and_logdet_init(phi, mu=0.0, sigma2=1.0)
        # Note: quad_and_logdet_init with mu=0,sigma2=1 just to get P0_inv stably (quad/logdet ignored)
        # This is safe because P0_inv depends only on phi.

        one_P0inv_one = float(ones_p @ P0_inv @ ones_p)
        one_P0inv_yinit = float(ones_p @ P0_inv @ y_init)

        # prior: mu ~ N(mu0, c2)
        prec = (1.0 / cfg.c2) + (n_star * (c_curr**2) / sigma2) + (one_P0inv_one / sigma2)
        V_mu = 1.0 / prec
        m_mu = V_mu * ((cfg.mu0 / cfg.c2) + (c_curr * S / sigma2) + (one_P0inv_yinit / sigma2))

        mu = float(rng.normal(m_mu, np.sqrt(V_mu)))

        # ---------- Step 3: Gibbs for sigma2 | (kappa, mu, y)  (exact)
        # conditional residuals
        resid_star = (y_p - X @ phi) - c_curr * mu * ones_star
        ss_star = float(np.sum(resid_star ** 2))

        # initial block quadratic with V^{-1} = (1/sigma2) P0^{-1}, and log|V| contributes p*log(sigma2)
        # For the sigma2 conditional, we need d'P0^{-1}d term:
        d_init = (y_init - mu * ones_p)
        ss_init = float(d_init @ P0_inv @ d_init)

        # IG update: likelihood contributes sigma2^{-(n_star/2)} exp(-ss_star/(2 sigma2))
        # and initial MVN contributes sigma2^{-(p/2)} exp(-ss_init/(2 sigma2))
        an = cfg.a0 + 0.5 * (n_star + p)
        bn = cfg.b0_ig + 0.5 * (ss_star + ss_init)
        sigma2 = 1.0 / rng.gamma(an, 1.0 / bn)

        # ---------- save
        if it >= cfg.burn and ((it - cfg.burn) % cfg.thin == 0):
            kappa_samps[save_i] = kappa
            mu_samps[save_i] = mu
            sigma2_samps[save_i] = sigma2
            save_i += 1

    accept_rate = accept_count / max(total_proposals, 1)

    return dict(
        kappa_samples=kappa_samps,
        mu_samples=mu_samps,
        sigma2_samples=sigma2_samps,
        accept_rate=accept_rate,
    )
