"""
Test script: Run MS-AR Gibbs sampler and generate diagnostic plots.

Tests both fixed and estimated transition matrix approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
from attempt1_mcmc_msar import gibbs_msar, MSARConfig
from msar_diagnostic_plots import create_all_diagnostic_plots
from partial_autocorrelation import pacf_to_phi


def simulate_msar_data(T=500, K=2, p=2, seed=42):
    """
    Simulate MS-AR(p, K) data with known parameters.
    """
    rng = np.random.default_rng(seed)

    # True parameters
    # State 0: low mean, low persistence
    # State 1: high mean, high persistence
    true_kappa = np.array([
        [0.3, -0.1],  # state 0
        [0.6, -0.2],  # state 1
    ])
    true_phi = np.array([pacf_to_phi(true_kappa[k, :]) for k in range(K)])
    true_mu = np.array([-1.0, 2.0])
    true_sigma2 = np.array([0.5, 0.3])
    true_sigma = np.sqrt(true_sigma2)

    # Sticky transition matrix (95% chance of staying in same state)
    true_xi = np.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])

    print("=" * 60)
    print("SIMULATION SETUP")
    print("=" * 60)
    print(f"\nModel: MS-AR({p}, {K})")
    print(f"Sample size: T = {T}")
    print(f"\nTrue parameters:")
    for k in range(K):
        print(f"\nState {k}:")
        print(f"  κ_{k} = {true_kappa[k, :]}")
        print(f"  φ_{k} = {true_phi[k, :]}")
        print(f"  μ_{k} = {true_mu[k]:.3f}")
        print(f"  σ²_{k} = {true_sigma2[k]:.3f}")

    print(f"\nTransition matrix:")
    print(true_xi)
    print()

    # Simulate states
    s_true = np.zeros(T, dtype=int)
    s_true[0] = rng.choice(K)
    for t in range(1, T):
        s_true[t] = rng.choice(K, p=true_xi[s_true[t - 1], :])

    # Simulate observations
    y = np.zeros(T)
    for t in range(p, T):
        k = s_true[t]
        lags = np.array([y[t - j] for j in range(1, p + 1)])
        c_k = 1.0 - np.sum(true_phi[k, :])
        mean_t = c_k * true_mu[k] + true_phi[k, :] @ lags
        y[t] = rng.normal(mean_t, true_sigma[k])

    # Print state occupancy
    print(f"State occupancies:")
    for k in range(K):
        occ = np.sum(s_true == k) / T
        print(f"  State {k}: {occ:.2%} ({np.sum(s_true == k)}/{T} time points)")
    print()

    true_params = {
        'kappa': true_kappa,
        'phi': true_phi,
        'mu': true_mu,
        'sigma2': true_sigma2,
        'xi': true_xi,
    }

    return y, s_true, true_params


def print_posterior_summary(results, true_params, K, p, label=""):
    """Print posterior summaries, handling None true_params for real data"""

    burnin = len(results['mu_samples']) // 5

    print(f"\n{'=' * 60}")
    print(f"POSTERIOR SUMMARY {label}")
    print(f"{'=' * 60}")

    for k in range(K):
        print(f"\nState {k + 1}:")
        print("-" * 40)

        # Mean
        mu_samples = results['mu_samples'][burnin:, k]
        print(f"  μ_{k + 1}:")
        print(f"    Posterior mean: {mu_samples.mean():.3f}")
        print(f"    95% CI: [{np.percentile(mu_samples, 2.5):.3f}, {np.percentile(mu_samples, 97.5):.3f}]")

        # Only print true value if it exists
        if true_params is not None and true_params['mu'] is not None:
            true_val = true_params['mu'][k]
            print(f"    True value: {true_val:.3f}")

        # Variance
        sigma2_samples = results['sigma2_samples'][burnin:, k]
        print(f"  σ²_{k + 1}:")
        print(f"    Posterior mean: {sigma2_samples.mean():.3f}")
        print(f"    95% CI: [{np.percentile(sigma2_samples, 2.5):.3f}, {np.percentile(sigma2_samples, 97.5):.3f}]")

        if true_params is not None and true_params['sigma2'] is not None:
            true_val = true_params['sigma2'][k]
            print(f"    True value: {true_val:.3f}")

        # PACF
        print(f"  κ_{k + 1}:")
        for j in range(p):
            kappa_samples = results['kappa_samples'][burnin:, k, j]
            print(
                f"    κ_{j + 1}: {kappa_samples.mean():.3f} [{np.percentile(kappa_samples, 2.5):.3f}, {np.percentile(kappa_samples, 97.5):.3f}]",
                end="")

            if true_params is not None and true_params['kappa'] is not None:
                true_val = true_params['kappa'][k, j]
                print(f" (true: {true_val:.3f})")
            else:
                print()  # Just newline

    # Transition matrix
    print(f"\n{'=' * 60}")
    print("TRANSITION MATRIX:")
    print(f"{'=' * 60}")

    xi_mean = results['xi_samples'][burnin:].mean(axis=0)

    print("\nPosterior mean:")
    print("       ", "  ".join([f"State {j + 1}" for j in range(K)]))
    for i in range(K):
        row_str = f"State {i + 1}:"
        for j in range(K):
            row_str += f"  {xi_mean[i, j]:.3f}  "
        print(row_str)

    # Only print true transition matrix if it exists
    if true_params is not None and true_params['xi'] is not None:
        print("\nTrue transition matrix:")
        print("       ", "  ".join([f"State {j + 1}" for j in range(K)]))
        for i in range(K):
            row_str = f"State {i + 1}:"
            for j in range(K):
                row_str += f"  {true_params['xi'][i, j]:.3f}  "
            print(row_str)


def run_test_fixed_xi(y, s_true, true_params, T=500, K=2, p=2, n_iter=5000, save_plots=False):
    """
    Test with FIXED transition matrix.
    """
    print("\n" + "=" * 60)
    print("TEST 1: FIXED TRANSITION MATRIX")
    print("=" * 60)

    # Configure sampler with FIXED transition matrix
    cfg = MSARConfig(
        K=K,
        p=p,
        n_iter=n_iter,
        burn=50,
        thin=2,
        estimate_xi=False,  # FIXED
        alpha_beta=(2.0, 2.0),
        mu0=0.0,
        c2=1e6,
        a0=2.0,
        b0_ig=1.0,
        prop_sd_kappa=0.05,
        rng_seed=123,
    )

    # Run sampler with TRUE transition matrix (known)
    print("\nRunning MCMC with fixed ξ (true value)...")
    results_fixed = gibbs_msar(y=y, xi=true_params['xi'], cfg=cfg)

    # Print summaries
    print_posterior_summary(results_fixed, true_params, K, p, "(Fixed ξ)")

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    save_dir = '/Users/FreddieLewin/Desktop/dissertation/plots/msar_mcmc_plots/xi_fixed' if save_plots else None
    figs_fixed = create_all_diagnostic_plots(
        y=y,
        results=results_fixed,
        true_params=true_params,
        true_states=s_true,
        save_dir=save_dir
    )

    return results_fixed, figs_fixed


def run_test_estimated_xi(y, s_true, true_params, T=500, K=2, p=2, n_iter=5000, save_plots=False):
    """
    Test with ESTIMATED transition matrix.
    """
    print("\n" + "=" * 60)
    print("TEST 2: ESTIMATED TRANSITION MATRIX")
    print("=" * 60)

    # Set up Dirichlet prior for transition matrix
    # Option 1: Uniform prior
    # alpha_xi = np.ones((K, K))

    # Option 2: Sticky prior (encourages persistence) NEW INITIAL TRANSITION MATRIX
    alpha_xi = np.ones((K, K)) * 2.0  # Stronger off-diagonal
    np.fill_diagonal(alpha_xi, 20.0)  # Favor staying in same state

    print(f"\nDirichlet prior on transition matrix:")
    print(alpha_xi)
    print(f"  (Diagonal elements = {alpha_xi[0,0]:.1f} encourage state persistence)")

    # Configure sampler with ESTIMATED transition matrix
    cfg = MSARConfig(
        K=K,
        p=p,
        n_iter=n_iter,
        burn=50,
        thin=2,
        estimate_xi=True,  # ESTIMATE ξ
        alpha_xi=alpha_xi,
        alpha_beta=(2.0, 2.0),
        mu0=0.0,
        c2=1e6,
        a0=2.0,
        b0_ig=1.0,
        prop_sd_kappa=0.05,
        rng_seed=123,
    )

    # Run sampler (xi will be learned)
    print("\nRunning MCMC with estimated ξ...")
    results_estimated = gibbs_msar(y=y, xi=None, cfg=cfg)  # xi=None, will be initialized

    # Print summaries
    print_posterior_summary(results_estimated, true_params, K, p, "(Estimated ξ)")

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    save_dir = '/Users/FreddieLewin/Desktop/dissertation/MSAR_MCMC_testing/plots_HMC_MSAR' if save_plots else None
    figs_estimated = create_all_diagnostic_plots(
        y=y,
        results=results_estimated,
        true_params=true_params,
        true_states=s_true,
        save_dir=save_dir
    )

    return results_estimated, figs_estimated


def compare_results(results_fixed, results_estimated, true_params, K, p):
    """
    Compare fixed vs estimated transition matrix results.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: FIXED vs ESTIMATED TRANSITION MATRIX")
    print("=" * 60)

    print("\n1. AR Parameter Estimates (should be similar):")
    print("-" * 60)
    for k in range(K):
        print(f"\nState {k}:")

        # Compare μ
        mu_fixed = np.mean(results_fixed['mu_samples'][:, k])
        mu_estimated = np.mean(results_estimated['mu_samples'][:, k])
        mu_true = true_params['mu'][k]
        print(f"  μ_{k}: Fixed={mu_fixed:.3f}, Estimated={mu_estimated:.3f}, True={mu_true:.3f}")

        # Compare σ²
        s2_fixed = np.mean(results_fixed['sigma2_samples'][:, k])
        s2_estimated = np.mean(results_estimated['sigma2_samples'][:, k])
        s2_true = true_params['sigma2'][k]
        print(f"  σ²_{k}: Fixed={s2_fixed:.3f}, Estimated={s2_estimated:.3f}, True={s2_true:.3f}")

    print("\n2. Transition Matrix:")
    print("-" * 60)
    print("True:")
    print(true_params['xi'])

    print("\nFixed (known):")
    print(true_params['xi'])

    if results_estimated['xi_samples'] is not None:
        xi_est_mean = np.mean(results_estimated['xi_samples'], axis=0)
        print("\nEstimated (posterior mean):")
        print(xi_est_mean)

        print("\nDifference (Estimated - True):")
        print(xi_est_mean - true_params['xi'])

        print("\n3. Transition Matrix Uncertainty:")
        print("-" * 60)
        for i in range(K):
            for j in range(K):
                samples = results_estimated['xi_samples'][:, i, j]
                mean = np.mean(samples)
                std = np.std(samples)
                ci = np.percentile(samples, [2.5, 97.5])
                true_val = true_params['xi'][i, j]
                print(f"  ξ_{i}{j}: {mean:.4f} ± {std:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], True: {true_val:.4f}")

    print("\n" + "=" * 60)


def run_complete_test(T=500, K=2, p=2, n_iter=5000, save_plots=True, show_plots=True):
    """
    Complete test: simulate data, run both fixed and estimated ξ versions, compare.
    """
    # Simulate data once
    y, s_true, true_params = simulate_msar_data(T=T, K=K, p=p, seed=42)

    # Test 1: Fixed transition matrix
    results_fixed, figs_fixed = run_test_fixed_xi(
        y, s_true, true_params, T, K, p, n_iter, save_plots
    )

    # Test 2: Estimated transition matrix
    results_estimated, figs_estimated = run_test_estimated_xi(
        y, s_true, true_params, T, K, p, n_iter, save_plots
    )

    # Compare results
    compare_results(results_fixed, results_estimated, true_params, K, p)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)

    if show_plots:
        print("\nClose the plot windows to exit.")
        plt.show()

    return {
        'data': (y, s_true, true_params),
        'fixed': (results_fixed, figs_fixed),
        'estimated': (results_estimated, figs_estimated),
    }


def run_single_test(mode='fixed', T=500, K=2, p=2, n_iter=5000, save_plots=False):
    """
    Run a single test (either 'fixed' or 'estimated').

    Args:
        mode: 'fixed' or 'estimated'
        T, K, p: model dimensions
        n_iter: number of MCMC iterations
        save_plots: whether to save plots to disk
    """
    # Simulate data
    y, s_true, true_params = simulate_msar_data(T=T, K=K, p=p, seed=42)

    if mode == 'fixed':
        results, figs = run_test_fixed_xi(y, s_true, true_params, T, K, p, n_iter, save_plots)
    elif mode == 'estimated':
        results, figs = run_test_estimated_xi(y, s_true, true_params, T, K, p, n_iter, save_plots)
    else:
        raise ValueError(f"mode must be 'fixed' or 'estimated', got {mode}")

    print("\nClose the plot windows to exit.")
    plt.show()

    return y, s_true, true_params, results, figs


def relabel_by_mean(results):
    """
    Relabel states so that μ₀ < μ₁ < ... < μₖ at each iteration.

    This fixes label switching issues.

    Args:
        results: dict from gibbs_msar (will be modified in place)

    Returns:
        results: same dict with relabeled samples
    """
    n_save, K = results['mu_samples'].shape
    p = results['kappa_samples'].shape[2]

    print("Relabeling states by mean...")

    for i in range(n_save):
        # Get ordering: indices that would sort mu in ascending order
        order = np.argsort(results['mu_samples'][i, :])

        # Relabel all parameters
        results['mu_samples'][i, :] = results['mu_samples'][i, order]
        results['sigma2_samples'][i, :] = results['sigma2_samples'][i, order]
        results['kappa_samples'][i, :, :] = results['kappa_samples'][i, order, :]

        # Relabel transition matrix (if estimated)
        if results['xi_samples'] is not None:
            # Relabel both rows and columns
            results['xi_samples'][i, :, :] = results['xi_samples'][i, order, :][:, order]

        # Relabel states (map old state labels to new ones)
        # Create mapping: old_label -> new_label
        mapping = np.argsort(order)  # inverse permutation
        old_states = results['state_samples'][i, :].copy()
        for t in range(len(old_states)):
            if old_states[t] >= 0:  # only relabel valid states
                results['state_samples'][i, t] = mapping[old_states[t]]

    print(f"Relabeled {n_save} iterations.")
    return results


def analyze_msar_results(
        y,
        dates,
        results,
        K,
        p,
        true_params=None,
        s_true=None,
        data_name="VIX",
        save_dir=None,
        show_plots=True,
        burnin_frac=0.5,
        major_events=None
):
    """
    Complete MS-AR analysis and visualization pipeline.

    Args:
        y: observed time series (array)
        dates: datetime index for observations (can be None for simulated data)
        results: dict from gibbs_msar with keys:
                 'mu_samples', 'kappa_samples', 'sigma2_samples',
                 'xi_samples', 'state_samples'
        K: number of states
        p: AR order
        true_params: dict with 'mu', 'kappa', 'sigma2', 'xi' (for simulated data, optional)
        s_true: true state sequence (for simulated data, optional)
        data_name: string identifier for the data (e.g., "VIX", "Simulated")
        save_dir: directory to save plots and results (None = don't save)
        show_plots: whether to display plots
        burnin_frac: fraction of samples to discard as burn-in (default 0.5)
        major_events: dict of {date_string: label} for marking events on plots
                      (only used if dates is not None)

    Returns:
        dict with:
            'smoothed_probs': state probabilities (T, K)
            'most_probable_states': mode state sequence (T,)
            'xi_mean': posterior mean transition matrix (K, K)
            'summaries': dict of posterior summaries
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    # ========================================
    # 0. SETUP
    # ========================================
    T = len(y)
    n_samples = len(results['mu_samples'])
    burnin = int(burnin_frac * n_samples)

    # If dates not provided, create integer index
    if dates is None:
        dates = np.arange(T)
        has_dates = False
    else:
        has_dates = True

    # Create save directories if needed
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'plots').mkdir(exist_ok=True)
        (save_dir / 'results').mkdir(exist_ok=True)

    # State labels
    if K == 2:
        state_labels = ['Low Vol', 'High Vol']
        colors = ['blue', 'red']
    elif K == 3:
        state_labels = ['Low Vol', 'Medium Vol', 'High Vol']
        colors = ['blue', 'orange', 'red']
    else:
        state_labels = [f'State {k + 1}' for k in range(K)]
        colors = [f'C{k}' for k in range(K)]

    print("=" * 60)
    print(f"MS-AR({p},{K}) ANALYSIS: {data_name}")
    print("=" * 60)
    print(f"\nSample size: {T}")
    print(f"MCMC samples: {n_samples} (burn-in: {burnin})")
    if has_dates:
        print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Data range: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

    # ========================================
    # 1. POSTERIOR SUMMARIES
    # ========================================
    print("\n" + "=" * 60)
    print("POSTERIOR PARAMETER ESTIMATES")
    print("=" * 60)

    summaries = {}

    for k in range(K):
        print(f"\n{state_labels[k].upper()} (State {k}):")
        print("-" * 40)

        # Mean
        mu_samples = results['mu_samples'][burnin:, k]
        mu_mean = mu_samples.mean()
        mu_ci = np.percentile(mu_samples, [2.5, 97.5])
        print(f"  Mean (μ_{k}):")
        print(f"    Posterior: {mu_mean:.3f} ± {mu_samples.std():.3f}")
        print(f"    95% CI: [{mu_ci[0]:.3f}, {mu_ci[1]:.3f}]")
        if true_params is not None:
            print(f"    True: {true_params['mu'][k]:.3f}")

        summaries[f'mu_{k}'] = {'mean': mu_mean, 'std': mu_samples.std(), 'ci': mu_ci}

        # Variance
        sigma2_samples = results['sigma2_samples'][burnin:, k]
        sigma2_mean = sigma2_samples.mean()
        sigma2_ci = np.percentile(sigma2_samples, [2.5, 97.5])
        print(f"  Variance (σ²_{k}):")
        print(f"    Posterior: {sigma2_mean:.3f} ± {sigma2_samples.std():.3f}")
        print(f"    95% CI: [{sigma2_ci[0]:.3f}, {sigma2_ci[1]:.3f}]")
        if true_params is not None:
            print(f"    True: {true_params['sigma2'][k]:.3f}")

        summaries[f'sigma2_{k}'] = {'mean': sigma2_mean, 'std': sigma2_samples.std(), 'ci': sigma2_ci}

        # PACF
        print(f"  PACF (κ_{k}):")
        for j in range(p):
            kappa_samples = results['kappa_samples'][burnin:, k, j]
            kappa_mean = kappa_samples.mean()
            kappa_ci = np.percentile(kappa_samples, [2.5, 97.5])
            print(f"    κ_{k},{j + 1}: {kappa_mean:.3f} [{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]")
            if true_params is not None:
                print(f"            (True: {true_params['kappa'][k, j]:.3f})")

            summaries[f'kappa_{k}_{j}'] = {'mean': kappa_mean, 'std': kappa_samples.std(), 'ci': kappa_ci}

    # Transition matrix
    print("\n" + "=" * 60)
    print("TRANSITION MATRIX")
    print("=" * 60)

    xi_samples = results['xi_samples'][burnin:]
    xi_mean = xi_samples.mean(axis=0)

    print("\nPosterior mean:")
    header = "       " + "  ".join([f"State {j}" for j in range(K)])
    print(header)
    for i in range(K):
        row_str = f"State {i}:"
        for j in range(K):
            row_str += f"  {xi_mean[i, j]:.3f}  "
        print(row_str)

    if true_params is not None and 'xi' in true_params:
        print("\nTrue:")
        for i in range(K):
            row_str = f"State {i}:"
            for j in range(K):
                row_str += f"  {true_params['xi'][i, j]:.3f}  "
            print(row_str)

    # Persistence
    print("\nExpected duration in each state:")
    for k in range(K):
        duration = 1 / (1 - xi_mean[k, k])
        duration_samples = 1 / (1 - xi_samples[:, k, k])
        duration_ci = np.percentile(duration_samples, [2.5, 97.5])

        if has_dates:
            unit = "days"
        else:
            unit = "periods"

        print(f"  State {k}: {duration:.1f} {unit} (95% CI: [{duration_ci[0]:.1f}, {duration_ci[1]:.1f}])")

        summaries[f'duration_{k}'] = {'mean': duration, 'ci': duration_ci}

    # ========================================
    # 2. STATE SEQUENCE ANALYSIS
    # ========================================
    print("\n" + "=" * 60)
    print("STATE SEQUENCE ANALYSIS")
    print("=" * 60)

    # Compute smoothed state probabilities
    state_samples_post_burnin = results['state_samples'][burnin:]
    smoothed_probs = np.zeros((T, K))

    for t in range(T):
        states_at_t = state_samples_post_burnin[:, t]
        valid_states = states_at_t[states_at_t >= 0]

        if len(valid_states) > 0:
            for k in range(K):
                smoothed_probs[t, k] = np.mean(valid_states == k)
        else:
            smoothed_probs[t, :] = 1.0 / K

    # Most probable state
    most_probable_states = smoothed_probs.argmax(axis=1)

    # Time in each state
    print("\nTime spent in each state:")
    for k in range(K):
        count = (most_probable_states == k).sum()
        pct = 100 * count / T
        print(f"  {state_labels[k]} (State {k}): {count}/{T} ({pct:.1f}%)")

        summaries[f'occupancy_{k}'] = {'count': count, 'percent': pct}

    # Compare to true states if available
    if s_true is not None:
        print("\nTrue state occupancy:")
        for k in range(K):
            count = (s_true == k).sum()
            pct = 100 * count / T
            print(f"  State {k}: {count}/{T} ({pct:.1f}%)")

        # Classification accuracy
        accuracy = (most_probable_states == s_true).mean()
        print(f"\nClassification accuracy: {100 * accuracy:.1f}%")
        summaries['accuracy'] = accuracy

    # Identify high-volatility periods (highest state)
    if has_dates:
        print(f"\nSample {state_labels[-1]} periods:")
        highest_state_idx = K - 1
        high_vol_indices = np.where(most_probable_states == highest_state_idx)[0]

        if len(high_vol_indices) > 0:
            # Find contiguous periods
            breaks = np.where(np.diff(high_vol_indices) > 1)[0]
            periods_start = [high_vol_indices[0]]
            periods_end = []

            for b in breaks:
                periods_end.append(high_vol_indices[b])
                periods_start.append(high_vol_indices[b + 1])
            periods_end.append(high_vol_indices[-1])

            # Show significant periods (at least 5 days)
            shown = 0
            for start_idx, end_idx in zip(periods_start, periods_end):
                duration = end_idx - start_idx + 1
                if duration >= 5 and shown < 10:
                    start_date = dates[start_idx]
                    end_date = dates[end_idx]
                    avg_val = y[start_idx:end_idx + 1].mean()
                    max_val = y[start_idx:end_idx + 1].max()
                    print(f"  {start_date} to {end_date}: {duration} periods, "
                          f"avg={avg_val:.1f}, max={max_val:.1f}")
                    shown += 1

    # ========================================
    # 3. PLOTS
    # ========================================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Plot 1: Data with state probabilities
    print("  1. State probabilities over time...")
    fig1, axes = plt.subplots(K + 1, 1, figsize=(14, 2.5 * (K + 1)))

    # Top panel: observed data
    axes[0].plot(dates, y, linewidth=0.8, color='black', alpha=0.7)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title(f'{data_name} Data', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Mark major events if provided
    if has_dates and major_events is not None:
        for date_str, label in major_events.items():
            try:
                event_date = pd.to_datetime(date_str)
                if event_date in dates:
                    for ax in axes:
                        ax.axvline(x=event_date, color='red', linestyle='--',
                                   alpha=0.3, linewidth=1)
            except:
                pass

    # State probability panels
    for k in range(K):
        axes[k + 1].fill_between(dates, 0, smoothed_probs[:, k],
                                 color=colors[k], alpha=0.5)
        axes[k + 1].plot(dates, smoothed_probs[:, k], color=colors[k], linewidth=1)
        axes[k + 1].set_ylabel(f'Pr({state_labels[k]})', fontsize=10)
        axes[k + 1].set_ylim([0, 1])
        axes[k + 1].grid(True, alpha=0.3)
        axes[k + 1].legend([f'State {k}'], loc='upper right')

    axes[-1].set_xlabel('Time' if not has_dates else 'Date', fontsize=11)
    plt.tight_layout()

    if save_dir is not None:
        fig1.savefig(save_dir / 'plots' / 'state_probabilities.png',
                     dpi=300, bbox_inches='tight')
        print(f"     Saved: {save_dir / 'plots' / 'state_probabilities.png'}")

    # Plot 2: Data colored by most probable state
    print("  2. Identified regimes...")
    fig2, ax = plt.subplots(figsize=(14, 6))

    for k in range(K):
        mask = most_probable_states == k
        ax.scatter(dates[mask], y[mask], c=colors[k], s=2, alpha=0.6,
                   label=f'{state_labels[k]} (State {k})')

    if s_true is not None:
        # Add true states as background
        for k in range(K):
            mask = s_true == k
            ax.scatter(dates[mask], y[mask], c=colors[k], s=10, alpha=0.1,
                       marker='s', edgecolors='none')

    ax.set_xlabel('Time' if not has_dates else 'Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'{data_name} with Identified Regimes', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', markerscale=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        fig2.savefig(save_dir / 'plots' / 'identified_regimes.png',
                     dpi=300, bbox_inches='tight')
        print(f"     Saved: {save_dir / 'plots' / 'identified_regimes.png'}")

    # ========================================
    # 4. SAVE RESULTS
    # ========================================
    if save_dir is not None:
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        # Save numerical results
        print("  Saving numerical results...")
        np.savez(save_dir / 'results' / 'posterior_samples.npz',
                 mu=results['mu_samples'][burnin:],
                 kappa=results['kappa_samples'][burnin:],
                 sigma2=results['sigma2_samples'][burnin:],
                 xi=results['xi_samples'][burnin:],
                 states=results['state_samples'][burnin:],
                 smoothed_probs=smoothed_probs,
                 most_probable_states=most_probable_states,
                 dates=dates,
                 y=y)
        print(f"     Saved: {save_dir / 'results' / 'posterior_samples.npz'}")

        # Save summary text file
        print("  Saving summary report...")
        with open(save_dir / 'results' / 'summary_report.txt', 'w') as f:
            f.write(f"MS-AR({p},{K}) Analysis: {data_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Sample size: {T}\n")
            if has_dates:
                f.write(f"Date range: {dates[0]} to {dates[-1]}\n")
            f.write(f"MCMC samples: {n_samples} (burn-in: {burnin})\n\n")

            f.write("STATE PARAMETERS (Posterior Mean and 95% CI):\n")
            f.write("=" * 60 + "\n")

            for k in range(K):
                f.write(f"\n{state_labels[k]} (State {k}):\n")
                f.write("-" * 40 + "\n")

                mu_samples = results['mu_samples'][burnin:, k]
                mu_ci = np.percentile(mu_samples, [2.5, 97.5])
                f.write(f"  μ: {mu_samples.mean():.3f} ± {mu_samples.std():.3f} "
                        f"[{mu_ci[0]:.3f}, {mu_ci[1]:.3f}]\n")

                sigma2_samples = results['sigma2_samples'][burnin:, k]
                sigma2_ci = np.percentile(sigma2_samples, [2.5, 97.5])
                f.write(f"  σ²: {sigma2_samples.mean():.3f} ± {sigma2_samples.std():.3f} "
                        f"[{sigma2_ci[0]:.3f}, {sigma2_ci[1]:.3f}]\n")

                f.write(f"  PACF (κ):\n")
                for j in range(p):
                    kappa_samples = results['kappa_samples'][burnin:, k, j]
                    kappa_ci = np.percentile(kappa_samples, [2.5, 97.5])
                    f.write(f"    κ_{j + 1}: {kappa_samples.mean():.3f} "
                            f"[{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("TRANSITION MATRIX (Posterior Mean):\n")
            f.write("=" * 60 + "\n")
            f.write("       " + "  ".join([f"State {j}" for j in range(K)]) + "\n")
            for i in range(K):
                row_str = f"State {i}:"
                for j in range(K):
                    row_str += f"  {xi_mean[i, j]:.3f}  "
                f.write(row_str + "\n")

            f.write("\nExpected Duration:\n")
            for k in range(K):
                duration = 1 / (1 - xi_mean[k, k])
                f.write(f"  State {k}: {duration:.1f} periods\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("TIME IN EACH STATE:\n")
            f.write("=" * 60 + "\n")
            for k in range(K):
                count = (most_probable_states == k).sum()
                pct = 100 * count / T
                f.write(f"  {state_labels[k]}: {count} periods ({pct:.1f}%)\n")

        print(f"     Saved: {save_dir / 'results' / 'summary_report.txt'}")

    # ========================================
    # 5. DISPLAY PLOTS
    # ========================================
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    # ========================================
    # 6. RETURN RESULTS
    # ========================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return {
        'smoothed_probs': smoothed_probs,
        'most_probable_states': most_probable_states,
        'xi_mean': xi_mean,
        'summaries': summaries,
        'figures': {'state_probs': fig1, 'regimes': fig2}
    }

if __name__ == '__main__':
    import yfinance as yf

    #vix = yf.download('^VIX', start='2000-01-01', end='2024-12-31', progress=False)
    #y = vix['Close'].dropna().values
    #dates = vix['Close'].dropna().index

    y, s_true, true_params = simulate_msar_data(T=500, K=2, p=2)

    T = len(y)  # Use all available data
    K = 2  # Three states (low/medium/high volatility)
    p = 2  # AR(4) order for each state
    n_iter = 3000  # MCMC iterations
    save_plots = True

    print(f"\n2. Model specification:")
    print(f"   States (K): {K}")
    print(f"   AR order (p): {p}")
    print(f"   MCMC iterations: {n_iter}")
    print(f"\n3. Running MS-AR({p},{K}) Gibbs sampler...")
    print("   (This may take several minutes...)\n")
    #true_params = None
    # {
    #    'mu': None,
    #    'kappa': None,
    #    'sigma2': None,
    #    'xi': None
    # }

    results, figs = run_test_estimated_xi(
        y=y,
        s_true=None,
        true_params=true_params,
        T=T,
        K=K,
        p=p,
        n_iter=n_iter,
        save_plots=save_plots
    )

    analysis = analyze_msar_results(
            y=y,
            dates=None,
            results=results,
            K=2,
            p=2,
            data_name="VIX (2000-2024)",
            save_dir="/Users/FreddieLewin/Desktop/dissertation/MSAR_MCMC_testing/plots_mine_MSAR",
            show_plots=True,
            major_events={
                '2008-09-15': '2008 Crisis',
                '2020-03-16': 'COVID-19'
            }
        )
