from cmdstanpy import CmdStanModel
import numpy as np
from pathlib import Path


def relabel_online(state_samples, param_samples, K):
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

    # Initialize MPM estimate with first draw
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
    all_perms = [np.array(perm) for perm in permutations(range(K))]  # Convert to numpy arrays

    # Track permutations applied
    applied_perms = [list(range(K))]  # Identity permutation for first draw

    print(f"Starting online relabelling for {N_iter} iterations...")

    for j in range(1, N_iter):
        if j % 500 == 0:
            print(f"  Relabelling iteration {j}/{N_iter}...")

        current_states = state_samples[j]

        # Find best permutation: minimize -sum(I(perm[s_j[t]] == mpm[t]))
        # Equivalently: maximize sum(I(perm[s_j[t]] == mpm[t]))
        best_perm = None
        best_score = -1

        for perm in all_perms:
            # Count agreements (perm is now a numpy array)
            score = np.sum(perm[current_states] == mpm_estimate)
            if score > best_score:
                best_score = score
                best_perm = perm

        # Apply best permutation to states
        relabelled_states[j] = best_perm[current_states]

        # Apply best permutation to parameters
        for k in range(K):
            orig_k = np.where(best_perm == k)[0][0]  # Which original state maps to position k
            relabelled_params['kappa'][j, k] = param_samples['kappa'][j, orig_k]
            relabelled_params['mu'][j, k] = param_samples['mu'][j, orig_k]
            relabelled_params['sigma2'][j, k] = param_samples['sigma2'][j, orig_k]

            # Permute transition matrix rows and columns
            for k2 in range(K):
                orig_k2 = np.where(best_perm == k2)[0][0]
                relabelled_params['xi'][j, k, k2] = param_samples['xi'][j, orig_k, orig_k2]

        applied_perms.append(best_perm.tolist())

        # Update MPM estimate: most frequent state at each time point
        for t in range(T):
            state_counts = np.bincount(relabelled_states[:j + 1, t], minlength=K)
            mpm_estimate[t] = np.argmax(state_counts)

    print("Relabelling complete!")

    return relabelled_states, relabelled_params, applied_perms


def fit_msar_stan(y, K, p,
                  alpha_pacf=2.0, beta_pacf=2.0,
                  mu0=0.0, c2=1e6,
                  a0=2.0, b0_ig=1.0,
                  seed=42,
                  num_chains=4,
                  num_samples=1000,
                  num_warmup=1000):
    """
    Fit MS-AR model using Stan (CmdStanPy).
    """
    # Use the ORIGINAL Stan file (without relabelling in Stan)
    stan_path = Path('/Users/FreddieLewin/PycharmProjects/bayesian_inference_for_stationary_autoregressions/filtering_smoothing_lib/msar_mcmc.stan')

    # Prepare data (no use_relabelling needed)
    data = {
        "N": len(y),
        "K": K,
        "p": p,
        "y": y.tolist(),
        "alpha_pacf": alpha_pacf,
        "beta_pacf": beta_pacf,
        "mu0": mu0,
        "c2": c2,
        "a0": a0,
        "b0_ig": b0_ig,
    }

    # Compile model
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=str(stan_path))

    # Run sampler
    print(f"Running MCMC: {num_chains} chains, {num_samples} samples, {num_warmup} warmup...")
    fit = model.sample(
        data=data,
        chains=num_chains,
        iter_sampling=num_samples,
        iter_warmup=num_warmup,
        seed=seed,
        show_console=True
    )

    return fit


def extract_msar_results(fit, K, p, apply_relabelling=True):
    """
    Extract MS-AR results from CmdStanPy fit object.
    Optionally apply online relabelling algorithm.
    """
    import numpy as np

    # Get draws as DataFrame
    df = fit.draws_pd()

    print("\nFirst 30 columns from Stan output:")
    print(list(df.columns)[:30])

    N = len(df)  # Total samples across chains

    # Find z columns to determine T
    z_cols = [col for col in df.columns if col.startswith('z[') or col.startswith('z.')]
    T = len(z_cols)

    # Extract PACF coefficients
    kappa_samples = np.zeros((N, K, p))
    for k in range(1, K + 1):
        for j in range(1, p + 1):
            possible_names = [
                f"kappa[{k},{j}]",
                f"kappa.{k}.{j}",
                f"kappa.{k},{j}",
            ]
            col_name = next((name for name in possible_names if name in df.columns), None)
            if col_name is None:
                raise KeyError(f"Cannot find kappa column for state {k}, lag {j}")
            kappa_samples[:, k - 1, j - 1] = df[col_name].to_numpy()

    # Extract means
    mu_samples = np.zeros((N, K))
    for k in range(1, K + 1):
        possible_names = [f"mu[{k}]", f"mu.{k}"]
        col_name = next((name for name in possible_names if name in df.columns), None)
        if col_name is None:
            raise KeyError(f"Cannot find mu column for state {k}")
        mu_samples[:, k - 1] = df[col_name].to_numpy()

    # Extract variances
    sigma2_samples = np.zeros((N, K))
    for k in range(1, K + 1):
        possible_names = [f"sigma2[{k}]", f"sigma2.{k}"]
        col_name = next((name for name in possible_names if name in df.columns), None)
        if col_name is None:
            raise KeyError(f"Cannot find sigma2 column for state {k}")
        sigma2_samples[:, k - 1] = df[col_name].to_numpy()

    # Extract transition matrix
    xi_samples = np.zeros((N, K, K))
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            possible_names = [f"gamma[{i},{j}]", f"gamma.{i}.{j}", f"gamma.{i},{j}"]
            col_name = next((name for name in possible_names if name in df.columns), None)
            if col_name is None:
                raise KeyError(f"Cannot find gamma column for states {i},{j}")
            xi_samples[:, i - 1, j - 1] = df[col_name].to_numpy()

    # Extract states
    state_samples = np.zeros((N, T), dtype=int)
    for n in range(1, T + 1):
        possible_names = [f"z[{n}]", f"z.{n}"]
        col_name = next((name for name in possible_names if name in df.columns), None)
        if col_name is None:
            raise KeyError(f"Cannot find z column for time {n}")
        state_samples[:, n - 1] = df[col_name].to_numpy().astype(int) - 1  # 0-indexed

    # Package parameters for relabelling
    param_samples = {
        'kappa': kappa_samples,
        'mu': mu_samples,
        'sigma2': sigma2_samples,
        'xi': xi_samples
    }

    # Apply relabelling if requested
    if apply_relabelling:
        print("\nApplying online relabelling algorithm...")
        relabelled_states, relabelled_params, applied_perms = relabel_online(
            state_samples, param_samples, K
        )

        return {
            'kappa_samples': relabelled_params['kappa'],
            'mu_samples': relabelled_params['mu'],
            'sigma2_samples': relabelled_params['sigma2'],
            'xi_samples': relabelled_params['xi'],
            'state_samples': relabelled_states,
            'applied_permutations': applied_perms,
            'raw_kappa_samples': kappa_samples,
            'raw_mu_samples': mu_samples,
            'raw_sigma2_samples': sigma2_samples,
            'raw_state_samples': state_samples,
        }
    else:
        return {
            'kappa_samples': kappa_samples,
            'mu_samples': mu_samples,
            'sigma2_samples': sigma2_samples,
            'xi_samples': xi_samples,
            'state_samples': state_samples,
        }


def run_stan_msar_test(y, s_true, true_params, K=2, p=2,
                       num_chains=4, num_samples=1000, num_warmup=1000,
                       save_plots=False, apply_relabelling=True):
    from test_sample_msar import create_all_diagnostic_plots, print_posterior_summary
    """
    Test Stan MS-AR sampler with optional relabelling.
    """
    print("\n" + "=" * 60)
    print("STAN HMC TEST: ESTIMATED TRANSITION MATRIX")
    print("=" * 60)

    print(f"\nPriors (matching Gibbs sampler):")
    print(f"  PACF: Beta(2.0, 2.0) on transformed kappa")
    print(f"  Mean: Normal(0.0, 1e6)")
    print(f"  Variance: Inv-Gamma(2.0, 1.0)")
    print(f"  Transition matrix: Dirichlet (uniform on simplex)")

    print(f"\nRunning Stan HMC...")
    print(f"  Chains: {num_chains}")
    print(f"  Warmup: {num_warmup}")
    print(f"  Samples per chain: {num_samples}")
    print(f"  Total samples: {num_chains * num_samples}")
    print(f"  Relabelling: {'ENABLED' if apply_relabelling else 'DISABLED'}")
    print()

    fit = fit_msar_stan(
        y=y,
        K=K,
        p=p,
        alpha_pacf=2.0,
        beta_pacf=2.0,
        mu0=0.0,
        c2=1e6,
        a0=2.0,
        b0_ig=1.0,
        seed=42,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup
    )

    # Extract results with/without relabelling
    results_stan = extract_msar_results(fit, K, p, apply_relabelling=apply_relabelling)

    # Print summaries
    suffix = "(Stan HMC - Relabelled)" if apply_relabelling else "(Stan HMC - Raw)"
    print_posterior_summary(results_stan, true_params, K, p, suffix)

    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    save_dir = '/Users/FreddieLewin/Desktop/dissertation/MSAR_MCMC_testing/plots_HMC_MSAR' if save_plots else None
    figs_stan = create_all_diagnostic_plots(
        y=y,
        results=results_stan,
        true_params=true_params,
        true_states=s_true,
        save_dir=save_dir
    )

    return results_stan, figs_stan


if __name__ == "__main__":
    from test_sample_msar import simulate_msar_data

    # Simulate data
    y, s_true, true_params = simulate_msar_data(T=500, K=2, p=2)

    T = len(y)
    K = 2
    p = 2
    save_plots = True

    # Stan settings
    num_chains = 4
    num_warmup = 1000
    num_samples = 750

    print("\n" + "=" * 60)
    print("RUNNING STAN (HMC) SAMPLER WITH RELABELLING")
    print("=" * 60)

    # Fit Stan model
    fit_stan = fit_msar_stan(
        y=y,
        K=K,
        p=p,
        alpha_pacf=2.0,
        beta_pacf=2.0,
        mu0=0.0,
        c2=1e6,
        a0=2.0,
        b0_ig=1.0,
        seed=42,
        num_chains=num_chains,
        num_samples=num_samples,
        num_warmup=num_warmup
    )

    print("\n" + "=" * 60)
    print("STAN FIT SUMMARY")
    print("=" * 60)
    print(fit_stan.summary())

    # Extract results WITH relabelling
    results_relabelled = extract_msar_results(fit_stan, K, p, apply_relabelling=True)

    # Extract results WITHOUT relabelling for comparison
    results_raw = extract_msar_results(fit_stan, K, p, apply_relabelling=False)

    import matplotlib.pyplot as plt
    import numpy as np

    # Create comparison plot: before and after relabelling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # RAW (before relabelling)
    mu_0_raw = results_raw['mu_samples'][:, 0]
    mu_1_raw = results_raw['mu_samples'][:, 1]

    iterations = np.arange(len(mu_0_raw))

    ax1.plot(iterations, mu_0_raw, alpha=0.7, linewidth=0.8, color='blue', label='μ₀')
    ax1.plot(iterations, mu_1_raw, alpha=0.7, linewidth=0.8, color='red', label='μ₁')
    ax1.axhline(true_params['mu'][0], color='blue', linestyle='--', linewidth=2,
                alpha=0.5, label=f"True μ₀ = {true_params['mu'][0]}")
    ax1.axhline(true_params['mu'][1], color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f"True μ₁ = {true_params['mu'][1]}")
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Mean Value', fontsize=12)
    ax1.set_title('(a) Before Relabelling: Label Switching Visible',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # RELABELLED (after relabelling)
    mu_0_rel = results_relabelled['mu_samples'][:, 0]
    mu_1_rel = results_relabelled['mu_samples'][:, 1]

    ax2.plot(iterations, mu_0_rel, alpha=0.7, linewidth=0.8, color='blue', label='μ₀')
    ax2.plot(iterations, mu_1_rel, alpha=0.7, linewidth=0.8, color='red', label='μ₁')
    ax2.axhline(true_params['mu'][0], color='blue', linestyle='--', linewidth=2,
                alpha=0.5, label=f"True μ₀ = {true_params['mu'][0]}")
    ax2.axhline(true_params['mu'][1], color='red', linestyle='--', linewidth=2,
                alpha=0.5, label=f"True μ₁ = {true_params['mu'][1]}")
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Mean Value', fontsize=12)
    ax2.set_title('(b) After Relabelling: Consistent State Labels',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        '/Users/FreddieLewin/Desktop/dissertation/MSAR_MCMC_testing/plots_HMC_MSAR/label_switching/relabelling_comparison.png',
        dpi=300, bbox_inches='tight')
    plt.show()
