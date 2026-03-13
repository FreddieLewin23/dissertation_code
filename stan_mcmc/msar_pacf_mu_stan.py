from cmdstanpy import CmdStanModel
import numpy as np
from pathlib import Path


def relabel_online_old(state_samples, param_samples, K):
    """
    Online relabelling algorithm based on MPM estimate.

    Args:
        state_samples: (N_iter, T) array of state sequences
        param_samples: dict with keys 'kappa', 'mu', 'sigma2', 'xi'
                      each of shape (N_iter, K, ...)
        K: number of states

    Returns:
        relabelled_states: (N_iter, T) array
        relabelled_params: dict with same structure as param_samples
    """
    N_iter, T = state_samples.shape

    # Find first valid time point (skip initial p observations which are 0)
    first_valid_t = 0
    for t in range(T):
        if state_samples[0, t] != 0:  # 0 means undefined in Stan output
            first_valid_t = t
            break

    print(f"First valid state at t={first_valid_t}, skipping first {first_valid_t} observations")

    # Initialize MPM estimate with first draw (only valid states)
    mpm_estimate = state_samples[0].copy()

    # Storage for relabelled output
    relabelled_states = np.zeros_like(state_samples)
    relabelled_states[0] = state_samples[0]

    # Storage for relabelled parameters
    relabelled_params = {
        'kappa': np.zeros_like(param_samples['kappa']),
        'mu': np.zeros_like(param_samples['mu']),
        'sigma2': np.zeros_like(param_samples['sigma2']),
        'xi': np.zeros_like(param_samples['xi'])
    }

    # First iteration - no permutation
    relabelled_params['kappa'][0] = param_samples['kappa'][0]
    relabelled_params['mu'][0] = param_samples['mu'][0]
    relabelled_params['sigma2'][0] = param_samples['sigma2'][0]
    relabelled_params['xi'][0] = param_samples['xi'][0]

    # Generate all permutations for K states
    from itertools import permutations
    all_perms = [np.array(perm) for perm in permutations(range(K))]

    # Track permutations applied
    applied_perms = [list(range(K))]

    print(f"Starting online relabelling for {N_iter} iterations...")

    for j in range(1, N_iter):
        if j % 500 == 0:
            print(f"  Relabelling iteration {j}/{N_iter}...")

        current_states = state_samples[j]

        # Find best permutation: maximize agreement with MPM (only on valid states)
        best_perm = None
        best_score = -1

        for perm in all_perms:
            # Count agreements only for valid time points (t >= first_valid_t)
            valid_states = current_states[first_valid_t:]
            valid_mpm = mpm_estimate[first_valid_t:]
            score = np.sum(perm[valid_states] == valid_mpm)
            if score > best_score:
                best_score = score
                best_perm = perm

        # Apply best permutation to states
        # Keep first p states as 0 (undefined)
        relabelled_states[j, :first_valid_t] = 0
        relabelled_states[j, first_valid_t:] = best_perm[current_states[first_valid_t:]]

        # Apply best permutation to parameters
        for k in range(K):
            orig_k = np.where(best_perm == k)[0][0]
            relabelled_params['kappa'][j, k] = param_samples['kappa'][j, orig_k]
            relabelled_params['mu'][j, k] = param_samples['mu'][j, orig_k]
            relabelled_params['sigma2'][j, k] = param_samples['sigma2'][j, orig_k]

            # Permute transition matrix rows and columns
            for k2 in range(K):
                orig_k2 = np.where(best_perm == k2)[0][0]
                relabelled_params['xi'][j, k, k2] = param_samples['xi'][j, orig_k, orig_k2]

        applied_perms.append(best_perm.tolist())

        # Update MPM estimate: most frequent state at each time point (only valid states)
        for t in range(first_valid_t, T):
            valid_states_at_t = relabelled_states[:j + 1, t]
            # Filter out any remaining 0s (shouldn't happen, but defensive)
            valid_states_at_t = valid_states_at_t[valid_states_at_t >= 0]
            if len(valid_states_at_t) > 0:
                state_counts = np.bincount(valid_states_at_t, minlength=K)
                mpm_estimate[t] = np.argmax(state_counts)

    print("Relabelling complete!")
    print(f"Applied {len(set(tuple(p) for p in applied_perms))} unique permutations")

    return relabelled_states, relabelled_params, applied_perms


def relabel_online(state_samples, param_samples, K):
    """
    Online relabelling algorithm based on MPM estimate.

    States are 0-indexed: {0, 1, ..., K-1}, with -1 for undefined
    """
    N_iter, T = state_samples.shape

    # Find first valid time point
    first_valid_t = 0
    for t in range(T):
        if state_samples[0, t] >= 0:
            first_valid_t = t
            break

    print(f"\nRelabelling: first valid state at t={first_valid_t}")

    # Initialize MPM by sorting states by variance (low → high)
    # This anchors the algorithm to a specific mode
    sigma2_first = param_samples['sigma2'][0]
    sorted_states = np.argsort(sigma2_first)  # Indices that sort by variance

    # Create initial permutation: map sorted indices back to 0, 1, ..., K-1
    init_perm = np.zeros(K, dtype=int)
    for new_label, old_label in enumerate(sorted_states):
        init_perm[old_label] = new_label

    # Apply initial permutation to first iteration
    mpm_estimate = state_samples[0].copy()
    valid_mask_0 = state_samples[0] >= 0
    mpm_estimate[valid_mask_0] = init_perm[state_samples[0, valid_mask_0]]

    print(f"  Initial variance order: {sigma2_first}")
    print(f"  Sorted indices: {sorted_states}")
    print(f"  Initial permutation applied: {init_perm}")

    # Storage
    relabelled_states = np.full_like(state_samples, -1)
    relabelled_states[0] = mpm_estimate.copy()

    relabelled_params = {
        'kappa': np.zeros_like(param_samples['kappa']),
        'mu': np.zeros_like(param_samples['mu']),
        'sigma2': np.zeros_like(param_samples['sigma2']),
        'xi': np.zeros_like(param_samples['xi'])
    }

    # Apply initial permutation to first parameter sample
    for k in range(K):
        orig_k = sorted_states[k]
        relabelled_params['kappa'][0, k] = param_samples['kappa'][0, orig_k]
        relabelled_params['mu'][0, k] = param_samples['mu'][0, orig_k]
        relabelled_params['sigma2'][0, k] = param_samples['sigma2'][0, orig_k]
        for k2 in range(K):
            orig_k2 = sorted_states[k2]
            relabelled_params['xi'][0, k, k2] = param_samples['xi'][0, orig_k, orig_k2]

    # All permutations
    from itertools import permutations as all_permutations
    all_perms = [np.array(perm) for perm in all_permutations(range(K))]

    applied_perms = [init_perm.tolist()]
    swap_count = 0

    print(f"\nProcessing {N_iter} iterations...")

    for j in range(1, N_iter):
        if j % 500 == 0:
            recent = swap_count
            print(f"  Iteration {j}/{N_iter}, swaps so far: {swap_count}/{j}")

        current_states = state_samples[j]
        valid_mask = current_states >= 0

        if not np.any(valid_mask):
            # No valid states (shouldn't happen)
            relabelled_states[j] = current_states
            for key in relabelled_params:
                relabelled_params[key][j] = param_samples[key][j]
            applied_perms.append(list(range(K)))
            continue

        # Find best permutation
        best_perm = None
        best_score = -1

        for perm in all_perms:
            valid_current = current_states[valid_mask]
            valid_mpm = mpm_estimate[valid_mask]

            # Apply permutation
            permuted = perm[valid_current]
            score = np.sum(permuted == valid_mpm)

            if score > best_score:
                best_score = score
                best_perm = perm

        # Check if we're swapping
        is_identity = np.array_equal(best_perm, np.arange(K))
        if not is_identity:
            swap_count += 1

        # Apply permutation to states
        relabelled_states[j] = state_samples[j].copy()
        relabelled_states[j, valid_mask] = best_perm[state_samples[j, valid_mask]]

        # Apply permutation to parameters
        for k in range(K):
            orig_k = np.where(best_perm == k)[0][0]
            relabelled_params['kappa'][j, k] = param_samples['kappa'][j, orig_k]
            relabelled_params['mu'][j, k] = param_samples['mu'][j, orig_k]
            relabelled_params['sigma2'][j, k] = param_samples['sigma2'][j, orig_k]
            for k2 in range(K):
                orig_k2 = np.where(best_perm == k2)[0][0]
                relabelled_params['xi'][j, k, k2] = param_samples['xi'][j, orig_k, orig_k2]

        applied_perms.append(best_perm.tolist())

        # Update MPM
        for t in range(first_valid_t, T):
            valid_at_t = relabelled_states[:j + 1, t]
            valid_at_t = valid_at_t[valid_at_t >= 0]

            if len(valid_at_t) > 0:
                counts = np.bincount(valid_at_t, minlength=K)
                mpm_estimate[t] = np.argmax(counts)

    print(f"\n✓ Relabelling complete:")
    print(f"  Total swaps: {swap_count}/{N_iter - 1} ({100 * swap_count / (N_iter - 1):.1f}%)")
    print(f"  Unique permutations: {len(set(tuple(p) for p in applied_perms))}")

    # Final check: print relabelled means
    print(f"\n  Final relabelled posterior means:")
    print(
        f"    State 0: μ={relabelled_params['mu'][:, 0].mean():.3f}, σ²={relabelled_params['sigma2'][:, 0].mean():.3f}")
    print(
        f"    State 1: μ={relabelled_params['mu'][:, 1].mean():.3f}, σ²={relabelled_params['sigma2'][:, 1].mean():.3f}")

    return relabelled_states, relabelled_params, applied_perms


def fit_msar_stan(y, K, p,
                  alpha=2.0, beta=2.0,
                  mu0=0.0, c2=1e6,
                  a0=5.0, b0_ig=4.0,
                  alpha_xi=1.0,
                  seed=42,
                  num_chains=4,
                  num_samples=1000,
                  num_warmup=1000):
    """
    Fit MS-AR model using Stan (CmdStanPy) with AUGMENTED state space.
    """
    # Read Stan code
    stan_path = Path(__file__).with_name("msar_mcmc.stan")

    # FORCE recompilation
    print("Compiling Stan model (forced)...")
    model = CmdStanModel(
        stan_file=str(stan_path),
        force_compile=True  # ← ADD THIS!
    )

    # Prepare data
    data = {
        "n": len(y),
        "K": K,
        "p": p,
        "y": y.tolist(),
        "alpha": alpha,
        "beta": beta,
        "mu0": mu0,
        "c2": c2,
        "a0": a0,
        "b0_ig": b0_ig,
        "alpha_xi": [alpha_xi] * K,
    }

    print("Compiling Stan model (forced)...")
    model = CmdStanModel(stan_file=str(stan_path), force_compile=True)

    # ============================================
    # FORCE ALL CHAINS TO START IN SAME MODE
    # ============================================

    # Compute reasonable initial values from data
    y_mean = np.mean(y)
    y_std = np.std(y)

    # Initialize mu in ASCENDING order (satisfies ordered constraint)
    init_mu = [y_mean - y_std, y_mean + y_std]  # Ensures mu[1] < mu[2]

    # Single initialization dict
    init_dict = {
        'mu': init_mu,  # Ordered!
        'sigma': [y_std, y_std],
        'kappa': [[0.0] * p for _ in range(K)],
        'xi': [[1.0 / K] * K for _ in range(K)]  # Uniform transition matrix
    }

    print(f"\nInitializing ALL chains with:")
    print(f"  mu = {init_mu}")
    print(f"  sigma = {[y_std, y_std]}")
    print()

    # Run sampler with SAME initialization for ALL chains
    print(f"Running MCMC: {num_chains} chains, {num_samples} samples, {num_warmup} warmup...")
    fit = model.sample(
        data=data,
        chains=num_chains,
        iter_sampling=num_samples,
        iter_warmup=num_warmup,
        seed=seed,
        inits=init_dict,  # ← SAME INIT FOR ALL CHAINS
        show_console=True
    )

    return fit


def extract_msar_results(fit, K, p, apply_relabeling=True):
    """
    Extract MS-AR results from CmdStanPy fit object with AUGMENTED state space.

    Args:
        fit: CmdStanPy fit object
        K: number of states
        p: AR order
        apply_relabeling: whether to apply online relabeling algorithm
    """
    import numpy as np

    # Get draws as DataFrame
    df = fit.draws_pd()

    print("\nFirst 30 columns from Stan output:")
    print(list(df.columns)[:30])

    N = len(df)  # Total samples across chains

    # Determine T from states column
    state_cols = [col for col in df.columns if col.startswith('states[') or col.startswith('states.')]
    T = len(state_cols)

    # Extract PACF coefficients: kappa[k,j] or kappa.k.j
    kappa_samples = np.zeros((N, K, p))
    for k in range(1, K + 1):
        for j in range(1, p + 1):
            possible_names = [
                f"kappa[{k},{j}]",
                f"kappa.{k}.{j}",
            ]
            col_name = next((name for name in possible_names if name in df.columns), None)
            if col_name is None:
                raise KeyError(f"Cannot find kappa for state {k}, lag {j}. "
                               f"Available: {[c for c in df.columns if 'kappa' in c][:10]}")
            kappa_samples[:, k - 1, j - 1] = df[col_name].to_numpy()

    # Extract means: mu[k]
    mu_samples = np.zeros((N, K))
    for k in range(1, K + 1):
        possible_names = [f"mu[{k}]", f"mu.{k}"]
        col_name = next((name for name in possible_names if name in df.columns), None)
        if col_name is None:
            raise KeyError(f"Cannot find mu for state {k}")
        mu_samples[:, k - 1] = df[col_name].to_numpy()

    # Extract variances: sigma2[k]
    sigma2_samples = np.zeros((N, K))
    for k in range(1, K + 1):
        possible_names = [f"sigma2[{k}]", f"sigma2.{k}"]
        col_name = next((name for name in possible_names if name in df.columns), None)
        if col_name is None:
            raise KeyError(f"Cannot find sigma2 for state {k}")
        sigma2_samples[:, k - 1] = df[col_name].to_numpy()

    # Extract transition matrix: xi[i,j]
    xi_samples = np.zeros((N, K, K))
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            possible_names = [
                f"xi[{i},{j}]",
                f"xi.{i}.{j}",
            ]
            col_name = next((name for name in possible_names if name in df.columns), None)
            if col_name is None:
                # Try xi_mat if it exists
                possible_names = [f"xi_mat[{i},{j}]", f"xi_mat.{i}.{j}"]
                col_name = next((name for name in possible_names if name in df.columns), None)
            if col_name is None:
                raise KeyError(f"Cannot find xi for states {i},{j}")
            xi_samples[:, i - 1, j - 1] = df[col_name].to_numpy()

    # Extract states (from generated quantities Viterbi)
    state_samples = np.zeros((N, T), dtype=int)
    for t in range(1, T + 1):
        possible_names = [f"states[{t}]", f"states.{t}"]
        col_name = next((name for name in possible_names if name in df.columns), None)
        if col_name is None:
            raise KeyError(f"Cannot find states for time {t}")
        # Stan uses 1-indexed states, convert to 0-indexed
        state_samples[:, t - 1] = df[col_name].to_numpy().astype(int) - 1

    # Prepare param dict for relabeling
    param_samples = {
        'kappa': kappa_samples,
        'mu': mu_samples,
        'sigma2': sigma2_samples,
        'xi': xi_samples
    }

    # Apply relabeling if requested
    if apply_relabeling:
        print("\nApplying online relabeling algorithm...")
        state_samples, param_samples, applied_perms = relabel_online(
            state_samples, param_samples, K
        )
        print(f"Applied {len(set(tuple(p) for p in applied_perms))} unique permutations")

    return {
        'kappa_samples': param_samples['kappa'],
        'mu_samples': param_samples['mu'],
        'sigma2_samples': param_samples['sigma2'],
        'xi_samples': param_samples['xi'],
        'state_samples': state_samples,
    }


def run_stan_msar_test(y, s_true=None, true_params=None, K=2, p=2,
                       num_chains=4, num_samples=1000, num_warmup=1000,
                       save_plots=False, apply_relabeling=False,  # ← Changed default to False
                       save_dir=None):
    """
    Test Stan MS-AR sampler with augmented state space.

    Args:
        y: Time series data
        s_true: True state sequence (optional, for simulated data validation)
        true_params: True parameters (optional, for simulated data validation)
        apply_relabeling: whether to apply online relabeling (default False for real data)
        save_dir: Directory to save plots (if None, uses default or doesn't save)
    """
    from test_sample_msar import create_all_diagnostic_plots, print_posterior_summary

    print("\n" + "=" * 60)
    print("STAN HMC: AUGMENTED STATE SPACE MS-AR")
    print("=" * 60)

    print(f"\nPriors:")
    print(f"  PACF: Beta(2.0, 2.0) on transformed kappa")
    print(f"  Mean: Ordered constraint (identifiability)")
    print(f"  Variance: Inv-Gamma(2, 1)")
    print(f"  Transition matrix: Dirichlet(1, ..., 1) per row")

    print(f"\nRelabeling: {'ENABLED' if apply_relabeling else 'DISABLED'}")

    print(f"\nRunning Stan HMC...")
    print(f"  Chains: {num_chains}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples per chain: {num_samples}")
    print(f"  Total samples: {num_chains * num_samples}")
    print()

    fit = fit_msar_stan(
        y=y,
        K=K,
        p=p,
        alpha=2.0,
        beta=2.0,
        mu0=0.0,
        c2=1e6,
        a0=2.0,
        b0_ig=1.0,
        alpha_xi=1.0,
        seed=42,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup
    )

    # Extract results
    results_stan = extract_msar_results(fit, K, p, apply_relabeling=apply_relabeling)

    # Print summaries
    if true_params is not None:
        # Simulated data - compare to truth
        print_posterior_summary(results_stan, true_params, K, p, "(Stan HMC)")
    else:
        # Real data - just print estimates
        print("\n" + "=" * 60)
        print("POSTERIOR SUMMARY (Stan HMC)")
        print("=" * 60)

        for k in range(K):
            print(f"\nState {k + 1}:")
            print(f"  μ_{k + 1}:")
            mu_mean = results_stan['mu_samples'][:, k].mean()
            mu_std = results_stan['mu_samples'][:, k].std()
            mu_ci = np.percentile(results_stan['mu_samples'][:, k], [2.5, 97.5])
            print(f"    Posterior mean: {mu_mean:.4f}")
            print(f"    95% CI: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]")

            print(f"  σ²_{k + 1}:")
            sig2_mean = results_stan['sigma2_samples'][:, k].mean()
            sig2_std = results_stan['sigma2_samples'][:, k].std()
            sig2_ci = np.percentile(results_stan['sigma2_samples'][:, k], [2.5, 97.5])
            print(f"    Posterior mean: {sig2_mean:.4f}")
            print(f"    95% CI: [{sig2_ci[0]:.4f}, {sig2_ci[1]:.4f}]")

            print(f"  κ_{k + 1}:")
            for j in range(p):
                kappa_mean = results_stan['kappa_samples'][:, k, j].mean()
                kappa_ci = np.percentile(results_stan['kappa_samples'][:, k, j], [2.5, 97.5])
                print(f"    κ_{j + 1}: {kappa_mean:.3f} [{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]")

        print("\n" + "=" * 60)
        print("TRANSITION MATRIX:")
        print("=" * 60)
        xi_mean = results_stan['xi_samples'].mean(axis=0)
        print("\nPosterior mean:")
        print(f"        State 1  State 2")
        for i in range(K):
            print(f"State {i + 1}:  {xi_mean[i, 0]:.3f}    {xi_mean[i, 1]:.3f}")

        # Expected durations
        print("\nExpected duration in each state:")
        for k in range(K):
            dur = 1 / (1 - xi_mean[k, k])
            dur_samples = 1 / (1 - results_stan['xi_samples'][:, k, k])
            dur_ci = np.percentile(dur_samples, [2.5, 97.5])
            print(f"  State {k + 1}: {dur:.1f} periods (95% CI: [{dur_ci[0]:.1f}, {dur_ci[1]:.1f}])")

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    if save_dir is None and save_plots:
        save_dir = '/Users/FreddieLewin/Desktop/dissertation/MSAR_MCMC_testing/plots_HMC_MSAR'

    figs_stan = create_all_diagnostic_plots(
        y=y,
        results=results_stan,
        true_params=true_params,  # Will handle None internally
        true_states=s_true,  # Will handle None internally
        save_dir=save_dir
    )

    return results_stan, figs_stan

if __name__ == "__main__":
    # Full analysis
    '''
    from test_sample_msar import analyze_msar_results
    from test_sample_msar import simulate_msar_data
    import yfinance as yf

    sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31')
    sp500_close = sp500['Close'].dropna()
    dates = sp500_close.index
    y = 100 * np.diff(np.log(sp500['Close'])['^GSPC'])


    results_sp500, figs = run_stan_msar_test(
        y=y,
        s_true=None,  # ← No true states
        true_params=None,  # ← No true parameters
        K=2,
        p=1,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        save_plots=True,
        apply_relabeling=False)  # ← CRITICAL: False for real data with ordered constraint
        #save_dir='/Users/FreddieLewin/Desktop/dissertation/MSAR_MCMC_testing/sp500_stan_results_largerT_p2_K3'


    '''

    # Simulate data
    from test_sample_msar import simulate_msar_data, analyze_msar_results


    K = 2
    p = 2

    y, s_true, true_params = simulate_msar_data(T=500, K=K, p=p)


    # Stan settings
    num_chains = 4
    num_warmup = 500
    num_samples = 1000  # Per chain, so 4 × 1000 = 4000 total

    # Run with relabeling
    results_stan, figs_stan = run_stan_msar_test(
        y=y,
        s_true=s_true,
        true_params=true_params,
        K=K,
        p=p,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup,
        save_plots=True,
        apply_relabeling=False  # NEW: enable relabeling
    )
