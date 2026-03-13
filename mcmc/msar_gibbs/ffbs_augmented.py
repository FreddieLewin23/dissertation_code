"""
MS-AR(p, K) Gibbs Sampler with Switching Mean and State-Space Augmentation

Model:
    S_t | S_{t-1} ~ Markov(ξ)
    y_t | S_t=k, S_{t-1:t-p}, y_{t-1:t-p} ~ N(μ_k + Σ φ_{j,k}(y_{t-j} - μ_{S_{t-j}}), σ²_k)

Uses state-space augmentation to restore Y3 assumption as described in Section 4.4.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy import linalg
from typing import Optional, Tuple
from dataclasses import dataclass

from partial_autocorrelation import pacf_to_phi, phi_to_pacf


@dataclass
class MSARConfig:
    """Configuration for MS-AR MCMC sampler."""
    K: int  # number of states
    p: int  # AR order (same for all states)
    n_iter: int = 10000
    burn: int = 2000
    thin: int = 1

    # Priors for PACF: κ_j ~ Beta via x=(κ+1)/2
    alpha_beta: tuple = (2.0, 2.0)

    # Prior for state means: μ_k ~ N(μ0, c²)
    mu0: float = 0.0
    c2: float = 1e6

    # Prior for innovation variance: σ²_k ~ InvGamma(a0, b0_ig)
    a0: float = 2.0
    b0_ig: float = 1.0

    # Prior for transition matrix: ξ_{j·} ~ Dirichlet(α_ξ[j, :])
    alpha_xi: Optional[np.ndarray] = None

    # Whether to estimate ξ
    estimate_xi: bool = True

    # MH proposal for κ (in z-space)
    prop_sd_kappa: float = 0.01

    # RNG
    rng_seed: Optional[int] = 42


# ============================================================================
# Helper functions (unchanged)
# ============================================================================
def stationary_P0(phi: np.ndarray) -> np.ndarray:
    """
    Solve discrete Lyapunov equation: P = A P A' + Q
    where Q = e_1 e_1' (innovation variance = 1).
    Returns P0 such that V(κ,σ²) = σ² * P0.

    Args:
        phi: AR coefficients, shape (p,)

    Returns:
        P0: stationary covariance matrix (normalized by σ²), shape (p, p)
    """
    p = len(phi)
    A = companion_matrix(phi)
    Q = np.zeros((p, p), dtype=float)
    Q[0, 0] = 1.0
    P0 = linalg.solve_discrete_lyapunov(A, Q)
    return P0


def companion_matrix(phi: np.ndarray) -> np.ndarray:
    """
    Build companion matrix for AR(p) process.
    State: s_t = [x_t, x_{t-1}, ..., x_{t-p+1}]', where x_t = y_t - μ
    x_t = phi' [x_{t-1},...,x_{t-p}] + eps_t

    Args:
        phi: AR coefficients, shape (p,)

    Returns:
        A: companion matrix, shape (p, p)
    """
    p = len(phi)
    A = np.zeros((p, p), dtype=float)
    A[0, :] = phi
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    return A


def kappa_to_z(kappa: np.ndarray) -> np.ndarray:
    """Transform κ ∈ (-1,1) to z ∈ ℝ via z = κ/√(1-κ²)."""
    return kappa / np.sqrt(1.0 - kappa ** 2)


def z_to_kappa(z: np.ndarray) -> np.ndarray:
    """Transform z ∈ ℝ to κ ∈ (-1,1) via κ = z/√(1+z²)."""
    return z / np.sqrt(1.0 + z ** 2)


def c_from_phi(phi: np.ndarray) -> float:
    """Compute c(φ) = 1 - Σ φ_j."""
    return float(1.0 - np.sum(phi))


def _lagged_matrix_at_indices(y: np.ndarray, indices: np.ndarray, p: int):
    """Build lagged design matrix for AR(p) at specific time indices."""
    if len(indices) == 0:
        return np.empty((0, p), dtype=float)

    X = np.zeros((len(indices), p), dtype=float)
    for i, t in enumerate(indices):
        if t < p:
            raise ValueError(f"Index {t} < p={p}, cannot construct lags")
        X[i, :] = [y[t - j] for j in range(1, p + 1)]

    return X


# ============================================================================
# Augmented State Space Functions
# ============================================================================

def encode_augmented_state(s_curr: int, s_hist: np.ndarray, K: int) -> int:
    """
    Encode augmented state (s_curr, s_hist[0], ..., s_hist[p-1]) to single index.

    Args:
        s_curr: current state S_t ∈ {0, ..., K-1}
        s_hist: history [S_{t-1}, ..., S_{t-p}], shape (p,)
        K: number of states

    Returns:
        Augmented state index ∈ {0, ..., K^{p+1}-1}
    """
    p = len(s_hist)
    idx = s_curr
    for j in range(p):
        idx += s_hist[j] * (K ** (j + 1))
    return idx


def decode_augmented_state(aug_idx: int, K: int, p: int) -> Tuple[int, np.ndarray]:
    """
    Decode augmented state index to (s_curr, s_hist).

    Args:
        aug_idx: augmented state index ∈ {0, ..., K^{p+1}-1}
        K: number of states
        p: AR order

    Returns:
        s_curr: current state
        s_hist: history [S_{t-1}, ..., S_{t-p}], shape (p,)
    """
    s_curr = aug_idx % K
    s_hist = np.zeros(p, dtype=int)

    remaining = aug_idx // K
    for j in range(p):
        s_hist[j] = remaining % K
        remaining //= K

    return s_curr, s_hist


def build_augmented_transition_matrix(xi: np.ndarray, p: int) -> np.ndarray:
    """
    Build sparse augmented transition matrix from original K×K matrix.

    Augmented state: (S_t, S_{t-1}, ..., S_{t-p})
    From (i_0, i_1, ..., i_p) can only transition to (j, i_0, i_1, ..., i_{p-1}) for any j.

    Args:
        xi: original transition matrix, shape (K, K)
        p: AR order

    Returns:
        xi_aug: augmented transition matrix, shape (K^{p+1}, K^{p+1})
                Sparse: each row has exactly K non-zero entries
    """
    K = xi.shape[0]
    K_aug = K ** (p + 1)

    xi_aug = np.zeros((K_aug, K_aug))

    # For each augmented state
    for aug_from in range(K_aug):
        s_curr, s_hist = decode_augmented_state(aug_from, K, p)

        # Can transition to any new current state j
        for j in range(K):
            # New history: shift left, add current state
            new_hist = np.zeros(p, dtype=int)
            new_hist[0] = s_curr  # what was S_t becomes S_{t-1}
            if p > 1:
                new_hist[1:] = s_hist[:p - 1]  # shift the rest

            aug_to = encode_augmented_state(j, new_hist, K)
            xi_aug[aug_from, aug_to] = xi[s_curr, j]

    return xi_aug


def extract_original_states(aug_states: np.ndarray, K: int, p: int) -> np.ndarray:
    """
    Extract original state sequence from augmented states.

    Args:
        aug_states: augmented state indices, shape (T,), values in {-1, 0, ..., K^{p+1}-1}
                   -1 indicates invalid (t < p)
        K: number of states
        p: AR order

    Returns:
        states: original state sequence, shape (T,), values in {-1, 0, ..., K-1}
    """
    T = len(aug_states)
    states = np.full(T, -1, dtype=int)

    for t in range(T):
        if aug_states[t] >= 0:
            s_curr, _ = decode_augmented_state(aug_states[t], K, p)
            states[t] = s_curr

    return states


# ============================================================================
# Augmented FFBS
# ============================================================================

def msar_loglik_t_augmented(
        y: np.ndarray,
        t: int,
        aug_state: int,
        phi: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        K: int,
        p: int,
) -> float:
    """
    Compute log p(y_t | augmented_state, y_{t-1:t-p}) for switching mean model.

    Model: y_t = μ_k + Σ φ_{j,k}(y_{t-j} - μ_{k_{-j}}) + ε_t

    Args:
        y: time series, shape (T,)
        t: time index
        aug_state: augmented state index containing (k, k_{-1}, ..., k_{-p})
        phi: AR coefficients, shape (K, p)
        mu: state means, shape (K,)
        sigma2: state variances, shape (K,)
        K: number of states
        p: AR order

    Returns:
        log-likelihood (scalar)
    """
    # Decode augmented state to get current state and history
    k, s_hist = decode_augmented_state(aug_state, K, p)

    # Lags: [y_{t-1}, ..., y_{t-p}]
    lags = np.array([y[t - j] for j in range(1, p + 1)])

    # Mean-centered lags using state history
    centered_lags = np.array([lags[j] - mu[s_hist[j]] for j in range(p)])

    # Conditional mean: μ_k + Σ φ_{j,k}(y_{t-j} - μ_{S_{t-j}})
    mean = mu[k] + phi[k, :] @ centered_lags

    # Gaussian log-likelihood
    resid2 = (y[t] - mean) ** 2
    loglik = -0.5 * np.log(2.0 * np.pi * sigma2[k]) - 0.5 * resid2 / sigma2[k]

    return float(loglik)


def msar_forward_filter_augmented(
        y: np.ndarray,
        phi: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        xi_aug: np.ndarray,
        init_probs_aug: np.ndarray,
        K: int,
        p: int,
) -> dict:
    """
    Forward filter for augmented MS-AR model.

    Returns:
        dict with log_alpha, alpha (shape (T, K^{p+1})), log_marginal_lik
    """
    from scipy.special import logsumexp

    T = len(y)
    K_aug = K ** (p + 1)

    # Storage
    log_alpha = np.full((T, K_aug), np.nan)

    # Initialize at t=p
    log_alpha[p, :] = np.log(init_probs_aug + 1e-300)
    log_alpha[p, :] -= logsumexp(log_alpha[p, :])

    log_xi_aug = np.log(xi_aug + 1e-300)
    log_marginal = 0.0

    for t in range(p, T):
        # Compute likelihoods for all augmented states
        ll = np.zeros(K_aug)
        for aug_k in range(K_aug):
            ll[aug_k] = msar_loglik_t_augmented(y, t, aug_k, phi, mu, sigma2, K, p)

        if t == p:
            unnorm = log_alpha[p, :] + ll
        else:
            # Prediction step
            log_pred = logsumexp(
                log_alpha[t - 1, :][:, None] + log_xi_aug,
                axis=0
            )
            unnorm = log_pred + ll

        # Normalize
        c = logsumexp(unnorm)
        log_alpha[t, :] = unnorm - c
        log_marginal += c

    alpha = np.exp(log_alpha)

    return {
        "log_alpha": log_alpha,
        "alpha": alpha,
        "log_marginal_lik": float(log_marginal),
    }


def msar_backward_sample_augmented(
        log_alpha: np.ndarray,
        xi_aug: np.ndarray,
        p: int,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Backward sampling for augmented states.

    Returns:
        aug_states: shape (T,), augmented state indices (-1 for t < p)
    """
    from scipy.special import logsumexp

    T, K_aug = log_alpha.shape
    aug_states = np.full(T, -1, dtype=int)

    # Sample terminal augmented state
    probs_T = np.exp(log_alpha[T - 1, :] - logsumexp(log_alpha[T - 1, :]))
    aug_states[T - 1] = rng.choice(K_aug, p=probs_T)

    # Backward recursion
    for t in range(T - 2, p - 1, -1):
        j = aug_states[t + 1]
        logw = log_alpha[t, :] + np.log(xi_aug[:, j] + 1e-300)
        logw -= logsumexp(logw)
        probs = np.exp(logw)
        aug_states[t] = rng.choice(K_aug, p=probs)

    return aug_states

