functions {
  vector pacf_to_ar(vector kappa) {
    int p = num_elements(kappa);
    vector[p] phi_prev;
    vector[p] phi_curr;

    // m = 1
    phi_prev[1] = kappa[1];
    for (m in 2:p) {
      for (i in 1:(m-1)) {
        // NOTE MINUS SIGN (correct)
        phi_curr[i] = phi_prev[i] - kappa[m] * phi_prev[m - i];
      }
      phi_curr[m] = kappa[m];

      for (i in 1:m) phi_prev[i] = phi_curr[i];
    }

    return phi_prev;
  }
}
data {
  int<lower=1> n;
  int<lower=1> p;
  vector[n] y;

  // priors / hyperparams
  real<lower=0> alpha;
  real<lower=0> beta;

  real mu0;
  real<lower=0> c2;

  real<lower=0> a0;
  real<lower=0> b0_ig;
}
parameters {
  vector<lower=-1, upper=1>[p] kappa;  // PACF components
  real mu;
  real<lower=0> sigma;
}
transformed parameters {
  vector[p] phi = pacf_to_ar(kappa);
  real c_phi = 1.0 - sum(phi);
}
model {
  // prior on kappa: independent Beta on x=(kappa+1)/2
  for (j in 1:p) {
    target += beta_lpdf(0.5 * (kappa[j] + 1.0) | alpha, beta);
  }

  // prior on mu
  mu ~ normal(mu0, sqrt(c2));

  // prior on sigma^2: Inv-Gamma(a0, b0_ig)
  // Implemented as sigma^2 ~ inv_gamma => target += inv_gamma_lpdf(sigma^2 | a0, b0_ig) + log Jacobian
  target += inv_gamma_lpdf(square(sigma) | a0, b0_ig) + log(2.0 * sigma);

  // likelihood
  for (t in (p+1):n) {
    real mean_t = c_phi * mu;
    for (j in 1:p) {
      mean_t += phi[j] * y[t - j];
    }
    y[t] ~ normal(mean_t, sigma);
  }
}
generated quantities {
  real sigma2 = square(sigma);
}
