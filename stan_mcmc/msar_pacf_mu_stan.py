from cmdstanpy import CmdStanModel

import numpy as np
from pathlib import Path


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
    # Read Stan code
    stan_path = Path(__file__).with_name("msar_mcmc.stan")

    # Prepare data
    data = {
        "N": len(y),
        "K": K,
        "p": p,
        "y": y.tolist(),  # CmdStan needs lists, not numpy arrays
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
        show_console=True  # Shows progress
    )

    return fit


def extract_msar_results(fit, K, p):
    """
    Extract MS-AR results from CmdStanPy fit object.
    """
    import numpy as np

    # Get draws as DataFrame
    df = fit.draws_pd()

    # Print first few column names for debugging
    print("\nFirst 30 columns from Stan output:")
    print(list(df.columns)[:30])

    N = len(df)  # Total samples across chains

    # Find a z column to determine T
    z_cols = [col for col in df.columns if col.startswith('z[') or col.startswith('z.')]
    T = len(z_cols)

    # Extract PACF coefficients
    # Stan uses bracket notation: kappa[k,j] or kappa.k.j
    kappa_samples = np.zeros((N, K, p))
    for k in range(1, K + 1):
        for j in range(1, p + 1):
            # Try different naming conventions
            possible_names = [
                f"kappa[{k},{j}]",
                f"kappa.{k}.{j}",
                f"kappa.{k},{j}",
            ]

            col_name = None
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break

            if col_name is None:
                raise KeyError(f"Cannot find kappa column for state {k}, lag {j}. "
                               f"Tried: {possible_names}. "
                               f"Available kappa columns: {[c for c in df.columns if 'kappa' in c]}")

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

    return {
        'kappa_samples': kappa_samples,
        'mu_samples': mu_samples,
        'sigma2_samples': sigma2_samples,
        'xi_samples': xi_samples,
        'state_samples': state_samples,
    }

def run_stan_msar_test(y, s_true, true_params, K=2, p=2,
                       num_chains=4, num_samples=1000, num_warmup=1000,
                       save_plots=False):
    from test_sample_msar import create_all_diagnostic_plots, print_posterior_summary
    """
    Test Stan MS-AR sampler (equivalent to run_test_estimated_xi for Gibbs).
    """
    print("\n" + "=" * 60)
    print("STAN HMC TEST: ESTIMATED TRANSITION MATRIX")
    print("=" * 60)

    # Stan uses same priors - print for comparison
    print(f"\nPriors (matching Gibbs sampler):")
    print(f"  PACF: Beta(2.0, 2.0) on transformed kappa")
    print(f"  Mean: Normal(0.0, 1e6)")
    print(f"  Variance: Inv-Gamma(2.0, 1.0)")
    print(f"  Transition matrix: Dirichlet (uniform on simplex)")
    print(f"  Note: Stan uses implicit uniform prior on transition matrix")

    # Run Stan sampler
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

    # Extract results
    results_stan = extract_msar_results(fit, K, p)

    # Print summaries (using same function as Gibbs)
    print_posterior_summary(results_stan, true_params, K, p, "(Stan HMC)")

    # Generate diagnostic plots (using same function as Gibbs)
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
