"""
MS-AR(p, K) Gibbs Sampler with State-Specific Means and Transition Matrix Estimation

Model:
    S_t | S_{t-1} ~ Markov(ξ)
    y_t | S_t=k, y_{t-1:t-p} ~ N(μ_k + Σ φ_{j,k}(y_{t-j} - μ_k), σ²_k)

    ξ_{j·} ~ Dirichlet(e_{j1}, ..., e_{jK}) for each row j

This implements the Gibbs sampler described in Section 4.6 of the dissertation.
"""

import numpy as np
from scipy.stats import beta as beta_dist
from scipy import linalg
from typing import Optional
from dataclasses import dataclass

from partial_autocorrelation import pacf_to_phi, phi_to_pacf
from ffbs_v2 import msar_forward_filter, msar_loglik_t


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
    # NEW: If None, defaults to uniform (all 1.0). Shape should be (K, K).
    alpha_xi: Optional[np.ndarray] = None

    # Whether to estimate ξ (if False, it's treated as fixed)
    estimate_xi: bool = True  # NEW

    # MH proposal for κ (in z-space)
    prop_sd_kappa: float = 0.01

    # RNG
    rng_seed: Optional[int] = 42


def kappa_to_z(kappa: np.ndarray) -> np.ndarray:
    """Transform κ ∈ (-1,1) to z ∈ ℝ via z = κ/√(1-κ²)."""
    return kappa / np.sqrt(1.0 - kappa ** 2)


def z_to_kappa(z: np.ndarray) -> np.ndarray:
    """Transform z ∈ ℝ to κ ∈ (-1,1) via κ = z/√(1+z²)."""
    return z / np.sqrt(1.0 + z ** 2)


def _lagged_matrix_at_indices(y: np.ndarray, indices: np.ndarray, p: int):
    """
    Build lagged design matrix for AR(p) at specific time indices.

    Args:
        y: full time series, shape (T,)
        indices: time points to include, each must be >= p
        p: lag order

    Returns:
        X: shape (len(indices), p), where X[i, j] = y[indices[i] - (j+1)]
    """
    if len(indices) == 0:
        return np.empty((0, p), dtype=float)

    X = np.zeros((len(indices), p), dtype=float)
    for i, t in enumerate(indices):
        if t < p:
            raise ValueError(f"Index {t} < p={p}, cannot construct lags")
        X[i, :] = [y[t - j] for j in range(1, p + 1)]

    return X


def c_from_phi(phi: np.ndarray) -> float:
    """Compute c(φ) = 1 - Σ φ_j."""
    return float(1.0 - np.sum(phi))


def companion_matrix(phi: np.ndarray) -> np.ndarray:
    """
    Build companion matrix for AR(p) process.
    State: s_t = [x_t, x_{t-1}, ..., x_{t-p+1}]', where x_t = y_t - μ
    """
    p = len(phi)
    A = np.zeros((p, p), dtype=float)
    A[0, :] = phi
    if p > 1:
        A[1:, :-1] = np.eye(p - 1)
    return A


def stationary_P0(phi: np.ndarray) -> np.ndarray:
    """
    Solve discrete Lyapunov equation: P = A P A' + Q
    where Q = e_1 e_1' (innovation enters only first element).
    Returns P0 such that V(κ,σ²) = σ² * P0.
    """
    A = companion_matrix(phi)
    Q = np.zeros((len(phi), len(phi)), dtype=float)
    Q[0, 0] = 1.0
    P0 = linalg.solve_discrete_lyapunov(A, Q)
    return P0


def count_transitions(states: np.ndarray, K: int) -> np.ndarray:
    """
    Count transitions n_{jk} = #{t : S_{t-1}=j, S_t=k}.

    Args:
        states: shape (T,), state sequence with values in {0, ..., K-1}
                or -1 for invalid states (t < p)
        K: number of states

    Returns:
        n: shape (K, K), transition counts
           n[j, k] = number of transitions from state j to state k
    """
    n = np.zeros((K, K), dtype=int)

    for t in range(1, len(states)):
        s_prev = states[t - 1]
        s_curr = states[t]

        # Only count valid transitions (both states >= 0)
        if s_prev >= 0 and s_curr >= 0:
            n[s_prev, s_curr] += 1

    return n


def sample_transition_matrix(
    states: np.ndarray,
    K: int,
    alpha: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample transition matrix from Dirichlet posterior given state sequence.

    Args:
        states: shape (T,), sampled state sequence, values in {0, ..., K-1}
                states[t] for t < p should be -1 (invalid/unobserved)
        K: number of states
        alpha: shape (K, K), Dirichlet prior parameters
               alpha[j, k] is prior count for transition j → k
        rng: numpy random generator

    Returns:
        xi: shape (K, K), sampled transition matrix
            xi[j, k] = P(S_t = k | S_{t-1} = j)
    """
    # Count transitions
    n = count_transitions(states, K)  # shape (K, K)

    # Sample each row independently from Dirichlet posterior
    xi = np.zeros((K, K))
    for j in range(K):
        # Posterior parameters for row j
        posterior_alpha = alpha[j, :] + n[j, :]

        # Sample from Dirichlet
        xi[j, :] = rng.dirichlet(posterior_alpha)

    return xi


def msar_backward_sample(
        log_alpha: np.ndarray,
        xi: np.ndarray,
        p: int,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Backward sampling step of FFBS.

    Args:
        log_alpha: shape (T, K), log P(S_t=k | y_{1:t})
        xi: shape (K, K), transition matrix
        p: lag order
        rng: numpy Generator

    Returns:
        s: shape (T,), sampled states; s[t] = -1 for t < p
    """
    T, K = log_alpha.shape
    s = np.full(T, -1, dtype=int)

    # Sample terminal state
    from scipy.special import logsumexp
    probs_T = np.exp(log_alpha[T - 1, :] - logsumexp(log_alpha[T - 1, :]))
    s[T - 1] = rng.choice(K, p=probs_T)

    # Backward recursion
    for t in range(T - 2, p - 1, -1):
        j = s[t + 1]
        logw = log_alpha[t, :] + np.log(xi[:, j] + 1e-300)
        logw -= logsumexp(logw)
        probs = np.exp(logw)
        s[t] = rng.choice(K, p=probs)

    return s


def sample_ar_params_for_state_k(
        y: np.ndarray,
        state_indices: np.ndarray,
        p: int,
        kappa_curr: np.ndarray,
        mu_curr: float,
        sigma2_curr: float,
        cfg: MSARConfig,
        rng: np.random.Generator,
) -> dict:
    """
    One iteration of Gibbs/MH updates for AR(p) parameters of a single state.

    Given observations y[state_indices] where S_t = k, sample (κ_k, μ_k, σ²_k).

    This follows the logic from gibbs_ar_pacf_with_mean but adapted for
    a subset of observations.

    Returns:
        dict with keys: kappa, mu, sigma2, accept (bool)
    """
    if len(state_indices) == 0:
        # No observations in this state - return current values
        return {
            'kappa': kappa_curr,
            'mu': mu_curr,
            'sigma2': sigma2_curr,
            'accept': False,
        }

    # Extract observations assigned to this state
    y_k = y[state_indices]
    n_k = len(y_k)

    # Build lagged matrix
    X_k = _lagged_matrix_at_indices(y, state_indices, p)

    # For initial block: find earliest observation in this state
    # (we'll use conditional likelihood only for simplicity; exact likelihood
    #  would require tracking initial p observations per state, which is complex)
    # Here we use conditional likelihood on all n_k observations.

    alpha, beta = cfg.alpha_beta

    def log_prior_kappa(k: np.ndarray) -> float:
        x = 0.5 * (k + 1.0)
        return float(np.sum(beta_dist.logpdf(x, alpha, beta)))

    # Current state
    phi_curr = pacf_to_phi(kappa_curr)
    c_curr = c_from_phi(phi_curr)

    # ========================================
    # Step 1: MH update for κ_k (via z-space)
    # ========================================
    z_curr = kappa_to_z(kappa_curr)

    # Current log posterior (conditional likelihood only)
    resid_curr = (y_k - X_k @ phi_curr) - c_curr * mu_curr
    ll_curr = -0.5 * np.sum(resid_curr ** 2) / sigma2_curr
    lp_curr = log_prior_kappa(kappa_curr)
    # Jacobian term for κ parameterization in z-space
    log_jac_curr = -1.5 * np.sum(np.log(1.0 - kappa_curr ** 2))  # FIX: negative sign
    logt_curr = ll_curr + lp_curr + log_jac_curr

    # Propose in z-space
    z_prop = z_curr + rng.normal(0.0, cfg.prop_sd_kappa, size=p)
    kappa_prop = z_to_kappa(z_prop)
    phi_prop = pacf_to_phi(kappa_prop)
    c_prop = c_from_phi(phi_prop)

    # Proposed log posterior
    resid_prop = (y_k - X_k @ phi_prop) - c_prop * mu_curr
    ll_prop = -0.5 * np.sum(resid_prop ** 2) / sigma2_curr
    lp_prop = log_prior_kappa(kappa_prop)
    log_jac_prop = -1.5 * np.sum(np.log(1.0 - kappa_prop ** 2))  # FIX: negative sign
    logt_prop = ll_prop + lp_prop + log_jac_prop

    log_acc_ratio = logt_prop - logt_curr

    if np.log(rng.uniform()) < log_acc_ratio:
        kappa_curr = kappa_prop
        phi_curr = phi_prop
        c_curr = c_prop
        accept = True
    else:
        accept = False

    # ========================================
    # Step 2: Gibbs update for μ_k
    # ========================================
    # Residuals: r = y_k - X_k @ phi = c μ + ε
    r_k = y_k - X_k @ phi_curr
    S_k = np.sum(r_k)

    # Prior: μ ~ N(μ0, c²)
    # Likelihood (conditional): y_k ~ N(X φ + c μ, σ² I)
    # Posterior: μ | ... ~ N(m_mu, V_mu)
    prec_mu = (1.0 / cfg.c2) + (n_k * c_curr ** 2 / sigma2_curr)
    V_mu = 1.0 / prec_mu
    m_mu = V_mu * ((cfg.mu0 / cfg.c2) + (c_curr * S_k / sigma2_curr))

    mu_new = rng.normal(m_mu, np.sqrt(V_mu))

    # ========================================
    # Step 3: Gibbs update for σ²_k
    # ========================================
    resid_final = (y_k - X_k @ phi_curr) - c_curr * mu_new
    ss_k = np.sum(resid_final ** 2)

    # Posterior: σ² | ... ~ InvGamma(a_n, b_n)
    a_n = cfg.a0 + 0.5 * n_k
    b_n = cfg.b0_ig + 0.5 * ss_k
    sigma2_new = 1.0 / rng.gamma(a_n, 1.0 / b_n)

    return {
        'kappa': kappa_curr,
        'mu': float(mu_new),
        'sigma2': float(sigma2_new),
        'accept': accept,
    }


def gibbs_msar(
        y: np.ndarray,
        xi: Optional[np.ndarray] = None,  # CHANGED: now optional
        cfg: Optional[MSARConfig] = None,
) -> dict:
    """
    Gibbs sampler for MS-AR(p, K) model with state-specific means.

    Args:
        y: observed time series, shape (T,)
        xi: transition matrix, shape (K, K), rows sum to 1
            - If cfg.estimate_xi=False: xi is FIXED throughout MCMC (must be provided)
            - If cfg.estimate_xi=True: xi is initial value, will be estimated (optional)
        cfg: MSARConfig with K, p, MCMC settings, priors

    Returns:
        dict with:
            kappa_samples: shape (n_save, K, p)
            mu_samples: shape (n_save, K)
            sigma2_samples: shape (n_save, K)
            state_samples: shape (n_save, T), sampled state paths
            xi_samples: shape (n_save, K, K) if estimate_xi=True, else None
            accept_rates: shape (K,), MH acceptance rate per state
    """
    if cfg is None:
        raise ValueError("Must provide MSARConfig")

    rng = np.random.default_rng(cfg.rng_seed)
    y = np.asarray(y, float).ravel()
    T = len(y)
    K = cfg.K
    p = cfg.p

    # ========================================
    # Initialize transition matrix
    # ========================================
    if cfg.estimate_xi:
        # Initialize xi (use provided or uniform)
        if xi is None:
            xi = np.ones((K, K)) / K  # uniform transitions
        else:
            xi = np.asarray(xi, float).copy()  # copy so we can modify

        # Set up Dirichlet prior
        if cfg.alpha_xi is None:
            alpha_xi = np.ones((K, K))  # uniform prior
        else:
            alpha_xi = np.asarray(cfg.alpha_xi, float)
            if alpha_xi.shape != (K, K):
                raise ValueError(f"alpha_xi must have shape ({K}, {K})")
    else:
        # Fixed xi mode
        if xi is None:
            raise ValueError("Must provide xi when estimate_xi=False")
        xi = np.asarray(xi, float)
        if xi.shape != (K, K):
            raise ValueError(f"xi must have shape ({K}, {K})")
        if not np.allclose(xi.sum(axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1")

    # ========================================
    # Initialize AR parameters
    # ======================================== NEW CODE
    kappa = np.zeros((K, p))
    mu = np.zeros(K)
    sigma2 = np.ones(K)
    phi = np.zeros((K, p))

    # Initialize means at data quantiles
    quantiles = np.linspace(0.2, 0.8, K)
    for k in range(K):
        mu[k] = np.quantile(y, quantiles[k])
        sigma2[k] = np.var(y) / K  # Split variance across states
        # Start with small positive AR(1) coefficient
        kappa[k, 0] = 0.3 + 0.2 * k  # e.g., 0.3, 0.5 for K=2
        phi[k, :] = pacf_to_phi(kappa[k, :])
    #===========================================£

    print(f"Initialized state means: {mu}")
    print(f"Initialized state variances: {sigma2}")
    # Initial state probabilities (uniform, or could use stationary dist of xi)
    init_probs = np.ones(K) / K

    # ========================================
    # Storage
    # ========================================
    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.zeros((n_save, K, p))
    mu_samps = np.zeros((n_save, K))
    sigma2_samps = np.zeros((n_save, K))
    state_samps = np.zeros((n_save, T), dtype=int)

    # NEW: storage for xi samples
    if cfg.estimate_xi:
        xi_samps = np.zeros((n_save, K, K))
    else:
        xi_samps = None

    accept_counts = np.zeros(K, dtype=int)
    proposal_counts = np.zeros(K, dtype=int)

    save_idx = 0

    print(f"Starting MS-AR({p}, {K}) Gibbs sampler...")
    print(f"  Iterations: {cfg.n_iter} (burn: {cfg.burn}, thin: {cfg.thin})")
    print(f"  Data: T={T} observations")
    print(f"  Transition matrix: {'ESTIMATED' if cfg.estimate_xi else 'FIXED'}")
    print()

    # ========================================
    # MCMC Loop
    # ========================================
    for it in range(cfg.n_iter):

        # ----------------------------------------
        # BLOCK 1: Sample States S_{p:T-1} | y, θ, ξ
        # ----------------------------------------
        # Run forward filter (uses current xi)
        filt_result = msar_forward_filter(
            y=y,
            phi=phi,
            mu=mu,
            sigma2=sigma2,
            xi=xi,  # uses current value (fixed or from previous iteration)
            init_probs=init_probs,
        )

        # Backward sample states
        s = msar_backward_sample(
            log_alpha=filt_result['log_alpha'],
            xi=xi,
            p=p,
            rng=rng,
        )

        # ----------------------------------------
        # BLOCK 2: Sample Parameters θ_k | y, S for each state k
        # ----------------------------------------
        for k in range(K):
            # Find time points assigned to state k (only valid for t >= p)
            state_mask = (s[p:] == k)  # boolean array on s[p:]
            state_indices = np.where(state_mask)[0] + p  # convert to absolute indices

            proposal_counts[k] += 1

            # Sample (κ_k, μ_k, σ²_k) given data in state k
            result_k = sample_ar_params_for_state_k(
                y=y,
                state_indices=state_indices,
                p=p,
                kappa_curr=kappa[k, :],
                mu_curr=mu[k],
                sigma2_curr=sigma2[k],
                cfg=cfg,
                rng=rng,
            )

            # Update parameters
            kappa[k, :] = result_k['kappa']
            mu[k] = result_k['mu']
            sigma2[k] = result_k['sigma2']
            phi[k, :] = pacf_to_phi(kappa[k, :])

            if result_k['accept']:
                accept_counts[k] += 1

        # ----------------------------------------
        # BLOCK 3: Sample Transition Matrix ξ | S (NEW!)
        # ----------------------------------------
        if cfg.estimate_xi:
            xi = sample_transition_matrix(
                states=s,
                K=K,
                alpha=alpha_xi,
                rng=rng,
            )

        # ----------------------------------------
        # Save samples (after burn-in)
        # ----------------------------------------
        if it >= cfg.burn and (it - cfg.burn) % cfg.thin == 0:
            kappa_samps[save_idx, :, :] = kappa
            mu_samps[save_idx, :] = mu
            sigma2_samps[save_idx, :] = sigma2
            state_samps[save_idx, :] = s

            if cfg.estimate_xi:
                xi_samps[save_idx, :, :] = xi

            save_idx += 1

        # Progress report
        if (it + 1) % 1000 == 0:
            print(f"  Iteration {it + 1}/{cfg.n_iter}")

    # Compute acceptance rates
    accept_rates = accept_counts / np.maximum(proposal_counts, 1)

    print("\nSampling complete!")
    print(f"Acceptance rates by state: {accept_rates}")

    if cfg.estimate_xi:
        print(f"\nPosterior mean transition matrix:")
        print(np.mean(xi_samps, axis=0))

    print()

    return {
        'kappa_samples': kappa_samps,
        'mu_samples': mu_samps,
        'sigma2_samples': sigma2_samps,
        'state_samples': state_samps,
        'xi_samples': xi_samps,  # NEW: None if not estimated
        'accept_rates': accept_rates,
    }
