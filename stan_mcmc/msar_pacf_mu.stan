functions {
  // Convert PACF to AR coefficients using Levinson recursion
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

  // Encode augmented state to single index (1-indexed)
  int encode_aug_state(int s_curr, array[] int s_hist, int K) {
    int p = size(s_hist);
    int idx = s_curr;
    int K_power = K;

    for (j in 1:p) {
      idx += (s_hist[j] - 1) * K_power;
      K_power *= K;
    }
    return idx;
  }

  // Decode augmented state index to (s_curr, s_hist)
  array[] int decode_aug_state(int aug_idx, int K, int p) {
    array[p+1] int result;
    int remaining = aug_idx;

    result[1] = ((remaining - 1) % K) + 1;
    remaining = (remaining - result[1]) %/% K + 1;  // Use %/% for integer division

    for (j in 1:p) {
      result[j+1] = ((remaining - 1) % K) + 1;
      remaining = (remaining - result[j+1]) %/% K + 1;
    }
    return result;
  }

  // Build sparse augmented transition matrix
  matrix build_aug_transition(matrix xi, int K, int p) {
    int K_aug = 1;
    for (i in 1:(p+1)) K_aug *= K;

    matrix[K_aug, K_aug] xi_aug = rep_matrix(0.0, K_aug, K_aug);

    for (aug_from in 1:K_aug) {
      array[p+1] int decoded = decode_aug_state(aug_from, K, p);
      int s_curr = decoded[1];
      array[p] int s_hist;
      for (j in 1:p) s_hist[j] = decoded[j+1];

      for (j in 1:K) {
        array[p] int new_hist;
        new_hist[1] = s_curr;
        if (p > 1) {
          for (k in 2:p) new_hist[k] = s_hist[k-1];
        }
        int aug_to = encode_aug_state(j, new_hist, K);
        xi_aug[aug_from, aug_to] = xi[s_curr, j];
      }
    }
    return xi_aug;
  }
}

data {
  int<lower=1> n;
  int<lower=1> p;
  int<lower=2> K;
  vector[n] y;

  real<lower=0> alpha;
  real<lower=0> beta;
  real mu0;
  real<lower=0> c2;
  real<lower=0> a0;
  real<lower=0> b0_ig;
  vector<lower=0>[K] alpha_xi;
}

transformed data {
  int K_aug = 1;
  for (i in 1:(p+1)) K_aug *= K;

  print("Augmented state space size: K_aug = ", K_aug);
  print("Initializing forward algorithm at t = ", p+1);
}

parameters {
  ordered[K] mu;
  array[K] vector<lower=-1, upper=1>[p] kappa;
  vector<lower=0>[K] sigma;
  array[K] simplex[K] xi;
}

transformed parameters {
  array[K] vector[p] phi;

  for (k in 1:K) {
    phi[k] = pacf_to_ar(kappa[k]);
  }

  matrix[K, K] xi_mat;
  for (i in 1:K) {
    for (j in 1:K) {
      xi_mat[i, j] = xi[i][j];
    }
  }

  matrix[K_aug, K_aug] xi_aug = build_aug_transition(xi_mat, K, p);
}

model {
  // Priors
  for (k in 1:K) {
    for (j in 1:p) {
      target += beta_lpdf(0.5 * (kappa[k][j] + 1.0) | alpha, beta);
    }
  }

  mu ~ normal(mu0, sqrt(c2));

  for (k in 1:K) {
    target += -2.0 * (a0 + 1.0) * log(sigma[k]) - b0_ig / square(sigma[k]);
  }

  for (i in 1:K) {
    xi[i] ~ dirichlet(alpha_xi);
  }

  // Likelihood via Forward Algorithm
  {
    matrix[n, K_aug] log_alpha;

    // INITIALIZE at t=p+1 (first valid AR observation)
    for (aug_k in 1:K_aug) {
      array[p+1] int decoded = decode_aug_state(aug_k, K, p);
      int k = decoded[1];
      array[p] int k_hist;
      for (j in 1:p) k_hist[j] = decoded[j+1];

      // Compute mean for y[p+1] using y[p], y[p-1], ..., y[1]
      real mean_init = mu[k];
      for (j in 1:p) {
        mean_init += phi[k][j] * (y[p+1-j] - mu[k_hist[j]]);
      }

      real log_lik = normal_lpdf(y[p+1] | mean_init, sigma[k]);
      log_alpha[p+1, aug_k] = -log(K_aug) + log_lik;
    }

    // FORWARD RECURSION for t = p+2, ..., n
    for (t in (p+2):n) {
      for (aug_k in 1:K_aug) {
        array[p+1] int decoded = decode_aug_state(aug_k, K, p);
        int k = decoded[1];
        array[p] int k_hist;
        for (j in 1:p) k_hist[j] = decoded[j+1];

        real mean_t = mu[k];
        for (j in 1:p) {
          mean_t += phi[k][j] * (y[t-j] - mu[k_hist[j]]);
        }

        real log_lik = normal_lpdf(y[t] | mean_t, sigma[k]);

        vector[K_aug] log_pred;
        for (aug_j in 1:K_aug) {
          real xi_val = xi_aug[aug_j, aug_k];
          if (xi_val > 0) {
            log_pred[aug_j] = log_alpha[t-1, aug_j] + log(xi_val);
          } else {
            log_pred[aug_j] = negative_infinity();
          }
        }

        log_alpha[t, aug_k] = log_sum_exp(log_pred) + log_lik;
      }
    }

    target += log_sum_exp(log_alpha[n, :]);
  }
}

generated quantities {
  vector[K] sigma2;
  for (k in 1:K) {
    sigma2[k] = square(sigma[k]);
  }

  array[n] int aug_states;
  array[n] int states;

  // Viterbi algorithm
  {
    matrix[n, K_aug] log_delta;
    array[n, K_aug] int psi;

    // INITIALIZE at t=p+1
    for (aug_k in 1:K_aug) {
      array[p+1] int decoded = decode_aug_state(aug_k, K, p);
      int k = decoded[1];
      array[p] int k_hist;
      for (j in 1:p) k_hist[j] = decoded[j+1];

      real mean_init = mu[k];
      for (j in 1:p) {
        mean_init += phi[k][j] * (y[p+1-j] - mu[k_hist[j]]);
      }

      real log_lik = normal_lpdf(y[p+1] | mean_init, sigma[k]);
      log_delta[p+1, aug_k] = -log(K_aug) + log_lik;
    }

    // FORWARD PASS for t = p+2, ..., n
    for (t in (p+2):n) {
      for (aug_k in 1:K_aug) {
        array[p+1] int decoded = decode_aug_state(aug_k, K, p);
        int k = decoded[1];
        array[p] int k_hist;
        for (j in 1:p) k_hist[j] = decoded[j+1];

        real mean_t = mu[k];
        for (j in 1:p) {
          mean_t += phi[k][j] * (y[t-j] - mu[k_hist[j]]);
        }
        real log_lik = normal_lpdf(y[t] | mean_t, sigma[k]);

        real max_val = negative_infinity();
        int best_prev = 1;

        for (aug_j in 1:K_aug) {
          real xi_val = xi_aug[aug_j, aug_k];
          if (xi_val > 0) {
            real val = log_delta[t-1, aug_j] + log(xi_val);
            if (val > max_val) {
              max_val = val;
              best_prev = aug_j;
            }
          }
        }

        psi[t, aug_k] = best_prev;
        log_delta[t, aug_k] = max_val + log_lik;
      }
    }

    // BACKWARD PASS
    {
      real max_val = negative_infinity();
      int best_final = 1;
      for (aug_k in 1:K_aug) {
        if (log_delta[n, aug_k] > max_val) {
          max_val = log_delta[n, aug_k];
          best_final = aug_k;
        }
      }
      aug_states[n] = best_final;
    }

    // Traceback from t=n down to t=p+2
    for (t in 1:(n-p-1)) {
      int tt = n - t + 1;
      aug_states[tt-1] = psi[tt, aug_states[tt]];
    }

    // EXTRACT STATES
    // First p+1 observations are undefined
    for (t in 1:(p+1)) {
      states[t] = 0;
    }

    // Extract from t=p+2 onwards
    for (t in (p+2):n) {
      array[p+1] int decoded = decode_aug_state(aug_states[t], K, p);
      states[t] = decoded[1];
    }
  }
}
