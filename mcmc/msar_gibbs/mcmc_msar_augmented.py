import numpy as np
from scipy.stats import beta as beta_dist
from scipy import linalg
from typing import Optional, Tuple
from dataclasses import dataclass

from partial_autocorrelation import pacf_to_phi, phi_to_pacf
from ffbs_augmented import (
    MSARConfig,
    build_augmented_transition_matrix,
    msar_backward_sample_augmented,
    msar_forward_filter_augmented,
    extract_original_states,
    stationary_P0
)
from ffbs_augmented import (
    _lagged_matrix_at_indices,
    c_from_phi,
    kappa_to_z,
    z_to_kappa
)


def count_transitions(states: np.ndarray, K: int) -> np.ndarray:
    """Count transitions in ORIGINAL (not augmented) states."""
    n = np.zeros((K, K), dtype=int)

    for t in range(1, len(states)):
        s_prev = states[t - 1]
        s_curr = states[t]

        if s_prev >= 0 and s_curr >= 0:
            n[s_prev, s_curr] += 1

    return n


def sample_transition_matrix(
        states: np.ndarray,
        K: int,
        alpha: np.ndarray,
        rng: np.random.Generator,
) -> np.ndarray:
    """Sample original K×K transition matrix from Dirichlet posterior."""
    n = count_transitions(states, K)

    xi = np.zeros((K, K))
    for j in range(K):
        posterior_alpha = alpha[j, :] + n[j, :]
        xi[j, :] = rng.dirichlet(posterior_alpha)

    return xi


# ============================================================================
# Parameter sampling with SWITCHING MEAN likelihood
# ============================================================================

def sample_ar_params_for_state_k(
        y: np.ndarray,
        state_indices: np.ndarray,
        states: np.ndarray,  # Full state sequence
        mu_all: np.ndarray,  # All K state means
        k: int,  # Which state we're updating
        p: int,
        kappa_curr: np.ndarray,
        mu_curr: float,
        sigma2_curr: float,
        cfg: MSARConfig,
        rng: np.random.Generator,
) -> dict:
    """
    One iteration of Gibbs/MH updates for AR(p) parameters of a single state.
    Uses SWITCHING MEAN likelihood.
    """
    if len(state_indices) == 0:
        return {
            'kappa': kappa_curr,
            'mu': mu_curr,
            'sigma2': sigma2_curr,
            'accept': False,
        }

    n_k = len(state_indices)

    alpha, beta = cfg.alpha_beta

    def log_prior_kappa(kap: np.ndarray) -> float:
        x = 0.5 * (kap + 1.0)
        return float(np.sum(beta_dist.logpdf(x, alpha, beta)))

    def compute_switching_mean_residuals(phi, mu):
        """
        Compute residuals for switching mean model:
        e_t = y_t - μ_k - Σ φ_j(y_{t-j} - μ_{S_{t-j}})
        """
        resid = np.zeros(n_k)
        for i, t in enumerate(state_indices):
            # Get mean-centered lags using state history
            mean_centered_lags = np.array([
                y[t - j] - mu_all[states[t - j]]
                for j in range(1, p + 1)
            ])

            # Switching mean: y_t = μ_k + Σ φ_j(y_{t-j} - μ_{S_{t-j}}) + ε
            predicted = mu + phi @ mean_centered_lags
            resid[i] = y[t] - predicted

        return resid

    phi_curr = pacf_to_phi(kappa_curr)

    # ========================================
    # Step 1: MH update for κ_k (via z-space)
    # ========================================
    z_curr = kappa_to_z(kappa_curr)

    resid_curr = compute_switching_mean_residuals(phi_curr, mu_curr)
    ll_curr = -0.5 * np.sum(resid_curr ** 2) / sigma2_curr
    lp_curr = log_prior_kappa(kappa_curr)
    log_jac_curr = -1.5 * np.sum(np.log(1.0 - kappa_curr ** 2))
    logt_curr = ll_curr + lp_curr + log_jac_curr

    # Propose
    z_prop = z_curr + rng.normal(0.0, cfg.prop_sd_kappa, size=p)
    kappa_prop = z_to_kappa(z_prop)
    phi_prop = pacf_to_phi(kappa_prop)

    resid_prop = compute_switching_mean_residuals(phi_prop, mu_curr)
    ll_prop = -0.5 * np.sum(resid_prop ** 2) / sigma2_curr
    lp_prop = log_prior_kappa(kappa_prop)
    log_jac_prop = -1.5 * np.sum(np.log(1.0 - kappa_prop ** 2))
    logt_prop = ll_prop + lp_prop + log_jac_prop

    log_acc_ratio = logt_prop - logt_curr

    if np.log(rng.uniform()) < log_acc_ratio:
        kappa_curr = kappa_prop
        phi_curr = phi_prop
        accept = True
    else:
        accept = False

    # ========================================
    # Step 2: Gibbs update for μ_k
    # ========================================
    # For switching mean: y_t = μ_k + Σ φ_j(y_{t-j} - μ_{S_{t-j}}) + ε
    # Rearranging: y_t - Σ φ_j(y_{t-j} - μ_{S_{t-j}}) = μ_k + ε

    S_k = 0.0
    for t in state_indices:
        mean_centered_lags = np.array([
            y[t - j] - mu_all[states[t - j]]
            for j in range(1, p + 1)
        ])
        residual_without_mu = y[t] - phi_curr @ mean_centered_lags
        S_k += residual_without_mu

    # Prior: μ ~ N(μ0, c²)
    # Likelihood: y_t - Σ φ_j(...) ~ N(μ_k, σ²)
    prec_mu = (1.0 / cfg.c2) + (n_k / sigma2_curr)
    V_mu = 1.0 / prec_mu
    m_mu = V_mu * ((cfg.mu0 / cfg.c2) + (S_k / sigma2_curr))

    mu_new = rng.normal(m_mu, np.sqrt(V_mu))

    # ========================================
    # Step 3: Gibbs update for σ²_k
    # ========================================
    # Update mu_all with new value before computing residuals
    mu_all_updated = mu_all.copy()
    mu_all_updated[k] = mu_new

    ss_k = 0.0
    for t in state_indices:
        mean_centered_lags = np.array([
            y[t - j] - mu_all_updated[states[t - j]]
            for j in range(1, p + 1)
        ])
        predicted = mu_new + phi_curr @ mean_centered_lags
        ss_k += (y[t] - predicted) ** 2

    a_n = cfg.a0 + 0.5 * n_k
    b_n = cfg.b0_ig + 0.5 * ss_k
    sigma2_new = 1.0 / rng.gamma(a_n, 1.0 / b_n)

    return {
        'kappa': kappa_curr,
        'mu': float(mu_new),
        'sigma2': float(sigma2_new),
        'accept': accept,
    }




# ============================================================================
# Main Gibbs sampler
# ============================================================================

def gibbs_msar(
        y: np.ndarray,
        xi: Optional[np.ndarray] = None,
        cfg: Optional[MSARConfig] = None,
) -> dict:
    """
    Gibbs sampler for switching mean MS-AR with state-space augmentation.
    """
    if cfg is None:
        raise ValueError("Must provide MSARConfig")

    rng = np.random.default_rng(cfg.rng_seed)
    y = np.asarray(y, float).ravel()
    T = len(y)
    K = cfg.K
    p = cfg.p
    K_aug = K ** (p + 1)

    # Initialize transition matrix
    if cfg.estimate_xi:
        xi = np.ones((K, K)) / K if xi is None else np.asarray(xi, float).copy()
        alpha_xi = np.ones((K, K)) if cfg.alpha_xi is None else np.asarray(cfg.alpha_xi, float)
    else:
        if xi is None:
            raise ValueError("Must provide xi when estimate_xi=False")
        xi = np.asarray(xi, float)

    # Initialize AR parameters
    kappa = np.zeros((K, p))
    mu = np.zeros(K)
    sigma2 = np.ones(K)
    phi = np.zeros((K, p))

    quantiles = np.linspace(0.2, 0.8, K)
    for k in range(K):
        mu[k] = np.quantile(y, quantiles[k])
        sigma2[k] = np.var(y) / K
        kappa[k, 0] = 0.3 + 0.2 * k
        phi[k, :] = pacf_to_phi(kappa[k, :])

    # Initial augmented state probabilities (uniform)
    init_probs_aug = np.ones(K_aug) / K_aug

    # Storage
    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.zeros((n_save, K, p))
    mu_samps = np.zeros((n_save, K))
    sigma2_samps = np.zeros((n_save, K))
    state_samps = np.zeros((n_save, T), dtype=int)

    if cfg.estimate_xi:
        xi_samps = np.zeros((n_save, K, K))
    else:
        xi_samps = None

    accept_counts = np.zeros(K, dtype=int)
    proposal_counts = np.zeros(K, dtype=int)
    save_idx = 0

    print(f"Starting Augmented MS-AR({p}, {K}) Gibbs sampler...")
    print(f"  Augmented state space size: {K_aug}")
    print(f"  Transition matrix: {'ESTIMATED' if cfg.estimate_xi else 'FIXED'}")
    print()

    # MCMC Loop
    for it in range(cfg.n_iter):

        # BLOCK 1: Sample augmented states
        xi_aug = build_augmented_transition_matrix(xi, p)

        filt_result = msar_forward_filter_augmented(
            y=y,
            phi=phi,
            mu=mu,
            sigma2=sigma2,
            xi_aug=xi_aug,
            init_probs_aug=init_probs_aug,
            K=K,
            p=p,
        )

        aug_states = msar_backward_sample_augmented(
            log_alpha=filt_result['log_alpha'],
            xi_aug=xi_aug,
            p=p,
            rng=rng,
        )

        # Extract original states
        s = extract_original_states(aug_states, K, p)

        # BLOCK 2: Sample parameters for each state
        for k in range(K):
            state_mask = (s[p:] == k)
            state_indices = np.where(state_mask)[0] + p

            proposal_counts[k] += 1

            result_k = sample_ar_params_for_state_k(
                y=y,
                state_indices=state_indices,
                states=s,  # Full state sequence
                mu_all=mu,  # All state means
                k=k,  # Which state we're updating
                p=p,
                kappa_curr=kappa[k, :],
                mu_curr=mu[k],
                sigma2_curr=sigma2[k],
                cfg=cfg,
                rng=rng,
            )

            kappa[k, :] = result_k['kappa']
            mu[k] = result_k['mu']
            sigma2[k] = result_k['sigma2']
            phi[k, :] = pacf_to_phi(kappa[k, :])

            if result_k['accept']:
                accept_counts[k] += 1

        # BLOCK 3: Sample transition matrix (using extracted states)
        if cfg.estimate_xi:
            xi = sample_transition_matrix(
                states=s,
                K=K,
                alpha=alpha_xi,
                rng=rng,
            )

        # Save samples
        if it >= cfg.burn and (it - cfg.burn) % cfg.thin == 0:
            kappa_samps[save_idx, :, :] = kappa
            mu_samps[save_idx, :] = mu
            sigma2_samps[save_idx, :] = sigma2
            state_samps[save_idx, :] = s  # Save ORIGINAL states

            if cfg.estimate_xi:
                xi_samps[save_idx, :, :] = xi

            save_idx += 1

        if (it + 1) % 50 == 0:
            print(f"  Iteration {it + 1}/{cfg.n_iter}")

    accept_rates = accept_counts / np.maximum(proposal_counts, 1)

    print("\nSampling complete!")
    print(f"Acceptance rates by state: {accept_rates}")

    if cfg.estimate_xi:
        print(f"\nPosterior mean transition matrix:")
        print(np.mean(xi_samps, axis=0))

    return {
        'kappa_samples': kappa_samps,
        'mu_samples': mu_samps,
        'sigma2_samples': sigma2_samps,
        'state_samples': state_samps,  # Original K-state sequence
        'xi_samples': xi_samps,
        'accept_rates': accept_rates,
    }
