functions {
  // Levinson recursion: PACF -> AR coefficients
  vector pacf_to_ar(vector kappa) {
    int p = num_elements(kappa);
    vector[p] phi_prev;
    vector[p] phi_curr;

    phi_prev[1] = kappa[1];
    for (m in 2:p) {
      for (i in 1:(m-1)) {
        phi_curr[i] = phi_prev[i] - kappa[m] * phi_prev[m - i];
      }
      phi_curr[m] = kappa[m];
      for (i in 1:m) phi_prev[i] = phi_curr[i];
    }
    return phi_prev;
  }
}

data {
  int<lower=1> N;              // Number of observations
  int<lower=1> K;              // Number of states
  int<lower=1> p;              // AR order
  array[N] real y;             // Observations

  // Priors
  real<lower=0> alpha_pacf;    // Beta prior for PACF
  real<lower=0> beta_pacf;
  real mu0;                     // Prior mean for state means
  real<lower=0> c2;             // Prior variance for state means
  real<lower=0> a0;             // Inv-Gamma shape for innovation variance
  real<lower=0> b0_ig;          // Inv-Gamma scale for innovation variance
}

parameters {
  // State-specific AR parameters
  array[K] vector<lower=-1, upper=1>[p] kappa;  // PACF for each state
  vector[K] mu;                                   // Mean for each state
  vector<lower=0>[K] sigma;                       // Innovation SD for each state

  // Transition matrix (rows)
  array[K] simplex[K] gamma_arr;

  // Initial state distribution
  simplex[K] rho;
}

transformed parameters {
  // Build transition matrix
  matrix[K, K] gamma;
  for (k in 1:K) {
    gamma[k, ] = to_row_vector(gamma_arr[k]);
  }

  // Convert PACF to AR coefficients for each state
  array[K] vector[p] phi;
  array[K] real c_phi;
  for (k in 1:K) {
    phi[k] = pacf_to_ar(kappa[k]);
    c_phi[k] = 1.0 - sum(phi[k]);
  }

  // Compute log likelihoods in each possible state
  matrix[K, N] log_omega;
  for (n in 1:N) {
    for (k in 1:K) {
      if (n <= p) {
        // For initial observations, use flat likelihood (or could use stationary dist)
        log_omega[k, n] = 0;
      } else {
        // AR(p) likelihood: y_t = c*mu + sum phi_j * y_{t-j} + eps
        real mean_t = c_phi[k] * mu[k];
        for (j in 1:p) {
          mean_t += phi[k][j] * y[n - j];
        }
        log_omega[k, n] = normal_lpdf(y[n] | mean_t, sigma[k]);
      }
    }
  }
}

model {
  // Priors on PACF
  for (k in 1:K) {
    for (j in 1:p) {
      target += beta_lpdf(0.5 * (kappa[k][j] + 1.0) | alpha_pacf, beta_pacf);
    }
  }

  // Priors on state means
  mu ~ normal(mu0, sqrt(c2));

  // Priors on innovation variances (sigma^2 ~ Inv-Gamma)
  for (k in 1:K) {
    target += inv_gamma_lpdf(square(sigma[k]) | a0, b0_ig) + log(2.0 * sigma[k]);
  }

  // Prior on transition matrix: uniform over simplexes (implicit)

  // Prior on initial state: uniform (implicit)

  // Likelihood via forward algorithm
  target += hmm_marginal(log_omega, gamma, rho);
}

generated quantities {
  // Sample latent state sequence
  array[N] int<lower=1, upper=K> z = hmm_latent_rng(log_omega, gamma, rho);

  // Compute filtered state probabilities
  matrix[K, N] state_probs = hmm_hidden_state_prob(log_omega, gamma, rho);

  // Convert sigma to sigma2 for comparison with your code
  vector[K] sigma2 = square(sigma);
}
