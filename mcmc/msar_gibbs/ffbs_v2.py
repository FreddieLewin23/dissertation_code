import numpy as np
from scipy.special import logsumexp


def msar_loglik_t(y: np.ndarray, t: int, phi: np.ndarray, mu: np.ndarray,
                  sigma2: np.ndarray) -> np.ndarray:
    """
    Return log p(y_t | S_t=k, y_{t-1:t-p}) for all k with state-specific means.

    Model: y_t = mu_k + sum_{j=1}^p phi_{j,k} (y_{t-j} - mu_k) + eps_t
           where eps_t ~ N(0, sigma2_k)

    This is equivalent to:
           y_t = c_k * mu_k + sum_{j=1}^p phi_{j,k} * y_{t-j} + eps_t
           where c_k = 1 - sum_{j=1}^p phi_{j,k}

    Args:
        y: shape (T,), observed time series
        t: int, must satisfy t >= p
        phi: shape (K, p), phi[k, j] = coefficient on y_{t-(j+1)} in regime k
        mu: shape (K,), state-specific means
        sigma2: shape (K,), innovation variances

    Returns:
        loglik: shape (K,), log p(y_t | S_t=k, y_{t-1:t-p}) for each k
    """
    K, p = phi.shape
    if t < p:
        raise ValueError(f"Need t >= p so that lags y_{{t-1:t-p}} exist. Got t={t}, p={p}")
    if sigma2.shape != (K,):
        raise ValueError(f"sigma2 must have shape ({K},), got {sigma2.shape}")
    if mu.shape != (K,):
        raise ValueError(f"mu must have shape ({K},), got {mu.shape}")

    # Extract lags: [y_{t-1}, y_{t-2}, ..., y_{t-p}]
    lags = np.array([y[t - j] for j in range(1, p + 1)], dtype=float)  # shape (p,)

    # Compute conditional mean for each state k
    # mean_k = mu_k + sum_{j=1}^p phi_{j,k} (y_{t-j} - mu_k)
    #        = mu_k * (1 - sum_j phi_{j,k}) + sum_j phi_{j,k} * y_{t-j}
    #        = c_k * mu_k + phi_k @ lags
    mean = np.zeros(K, dtype=float)
    for k in range(K):
        c_k = 1.0 - np.sum(phi[k, :])  # same c(φ) coefficient as in single AR case
        mean[k] = c_k * mu[k] + phi[k, :] @ lags

    # Gaussian log-likelihood: log N(y_t; mean_k, sigma2_k)
    # = -0.5 * log(2π * sigma2_k) - 0.5 * (y_t - mean_k)^2 / sigma2_k
    resid2 = (y[t] - mean) ** 2
    loglik = -0.5 * np.log(2.0 * np.pi * sigma2) - 0.5 * resid2 / sigma2

    return loglik


def msar_forward_filter(
        y: np.ndarray,
        phi: np.ndarray,
        mu: np.ndarray,
        sigma2: np.ndarray,
        xi: np.ndarray,
        init_probs: np.ndarray,
) -> dict:
    """
    Forward filter for MS-AR(p, K) with state-specific means and fixed transition matrix.

    Model:
        S_t | S_{t-1} ~ Categorical(xi[S_{t-1}, :])
        y_t | S_t=k, y_{t-1:t-p} ~ N(mu_k + sum phi_{j,k}(y_{t-j} - mu_k), sigma2_k)

    Computes filtered probabilities:
        alpha_t(k) = P(S_t = k | y_{1:t}, theta)

    for t = p, p+1, ..., T-1.

    Args:
        y: shape (T,), observed time series
        phi: shape (K, p), autoregressive coefficients for each state
        mu: shape (K,), state-specific means
        sigma2: shape (K,), innovation variances for each state
        xi: shape (K, K), transition matrix where xi[i, j] = P(S_t=j | S_{t-1}=i)
                         Rows must sum to 1.
        init_probs: shape (K,), initial state probabilities P(S_p = k) before observing y_p
                    Often set to stationary distribution of xi, or uniform (1/K).

    Returns:
        dict with:
            log_alpha: shape (T, K), log filtered probabilities
                       log_alpha[t, k] = log P(S_t = k | y_{1:t})
                       Only valid for t >= p; entries for t < p are NaN.
            alpha: shape (T, K), filtered probabilities (exponentiated log_alpha)
                   alpha[t, k] = P(S_t = k | y_{1:t})
            log_marginal_lik: float, log p(y_{p:T-1} | theta)
                              Sum of log normalizing constants from filtering.
            p: int, lag order (for convenience in downstream code)

    Notes:
        - The filter starts at t=p (the first time point where all p lags are available).
        - log_alpha[t, :] and alpha[t, :] are NaN for t < p.
        - All computations done in log-space for numerical stability.
    """
    y = np.asarray(y, float).ravel()
    K, p = phi.shape
    T = y.shape[0]

    # Validate inputs
    xi = np.asarray(xi, float)
    if xi.shape != (K, K):
        raise ValueError(f"xi must have shape ({K}, {K}), got {xi.shape}")
    if not np.allclose(xi.sum(axis=1), 1.0):
        raise ValueError("Transition matrix xi rows must sum to 1")

    if mu.shape != (K,):
        raise ValueError(f"mu must have shape ({K},), got {mu.shape}")
    if sigma2.shape != (K,):
        raise ValueError(f"sigma2 must have shape ({K},), got {sigma2.shape}")
    if init_probs.shape != (K,):
        raise ValueError(f"init_probs must have shape ({K},), got {init_probs.shape}")
    if not np.allclose(init_probs.sum(), 1.0):
        raise ValueError("init_probs must sum to 1")

    # Work in log space for numerical stability
    log_xi = np.log(xi + 1e-300)  # avoid log(0); assumes no truly impossible transitions

    # Storage for filtered log probabilities
    log_alpha = np.full((T, K), np.nan, dtype=float)

    # Initialize at t=p with prior distribution over S_p
    # This is P(S_p = k) before we've seen any observations
    log_alpha[p, :] = np.log(init_probs + 1e-300)
    log_alpha[p, :] -= logsumexp(log_alpha[p, :])  # normalize (should already sum to 1)

    # Accumulator for log marginal likelihood
    # log p(y_{p:T-1} | theta) = sum_{t=p}^{T-1} log p(y_t | y_{1:t-1}, theta)
    log_marginal = 0.0

    # Forward recursion from t=p to T-1
    for t in range(p, T):
        # ----------------------------------------
        # Compute likelihood for y_t under each regime
        # ----------------------------------------
        # ll[k] = log p(y_t | S_t=k, y_{t-1:t-p}, theta)
        ll = msar_loglik_t(y, t, phi, mu, sigma2)  # shape (K,)

        if t == p:
            # ----------------------------------------
            # Special case: first usable observation at t=p
            # ----------------------------------------
            # We already have the prior log_alpha[p, :] = log P(S_p=k)
            # Update with likelihood: P(S_p=k | y_p) ∝ P(S_p=k) * p(y_p | S_p=k)
            unnorm = log_alpha[p, :] + ll
        else:
            # ----------------------------------------
            # General case: t > p
            # ----------------------------------------
            # Step 1: Prediction
            # P(S_t = k | y_{1:t-1}) = sum_{j=1}^K P(S_{t-1}=j | y_{1:t-1}) * xi[j, k]
            #
            # In log-space:
            # log P(S_t=k | y_{1:t-1}) = logsumexp_j [ log_alpha[t-1, j] + log_xi[j, k] ]
            #
            # Broadcasting: log_alpha[t-1, :] is shape (K,), log_xi is (K, K)
            # We want logsumexp over axis 0 (source states j)
            log_pred = logsumexp(
                log_alpha[t - 1, :][:, None] + log_xi,  # shape (K, K)
                axis=0  # sum over source states (rows)
            )  # shape (K,)

            # Step 2: Update with likelihood
            # P(S_t=k | y_{1:t}) ∝ P(S_t=k | y_{1:t-1}) * p(y_t | S_t=k, y_{t-1:t-p})
            unnorm = log_pred + ll

        # ----------------------------------------
        # Normalize to get filtered distribution
        # ----------------------------------------
        # c = sum_k exp(unnorm[k]) = p(y_t | y_{1:t-1}, theta)
        c = logsumexp(unnorm)
        log_alpha[t, :] = unnorm - c  # now sums to 1 in probability space

        # Accumulate log marginal likelihood
        log_marginal += c

    # Convert to probability space (will have NaNs for t < p, which is fine)
    alpha = np.exp(log_alpha)

    return {
        "log_alpha": log_alpha,
        "alpha": alpha,
        "log_marginal_lik": float(log_marginal),
        "p": p,
    }


if __name__ == "__main__":
    # Quick test with state-specific means
    print("Testing MS-AR forward filter with state-specific means...")

    # Simple 2-state AR(1) example
    K = 2
    p = 1
    T = 100

    phi = np.array([
        [0.3],  # state 0: φ = 0.3
        [0.7],  # state 1: φ = 0.7
    ])

    mu = np.array([2.0, -1.0])  # state-specific means
    sigma2 = np.array([0.5, 0.3])

    xi = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])

    init_probs = np.array([0.5, 0.5])

    # Simulate some data (just use random normal for quick test)
    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, size=T)

    # Run filter
    result = msar_forward_filter(
        y=y,
        phi=phi,
        mu=mu,
        sigma2=sigma2,
        xi=xi,
        init_probs=init_probs,
    )

    print(f" Filter ran successfully")
    print(f" Log marginal likelihood: {result['log_marginal_lik']:.4f}")
    print(f" Filtered probs at t=p:   {result['alpha'][p, :]}")
    print(f" Filtered probs at t=T-1: {result['alpha'][T - 1, :]}")
    print(f" Sum of alpha[T-1]:       {np.sum(result['alpha'][T - 1, :]):.6f} (should be 1.0)")

    # Check that probabilities sum to 1 at all valid time points
    for t in range(p, T):
        prob_sum = np.sum(result['alpha'][t, :])
        assert np.abs(prob_sum - 1.0) < 1e-10, f"Probabilities don't sum to 1 at t={t}"

    print(f"✓ All filtered probabilities sum to 1")
    print("\nTest passed!")
