import numpy as np

def phi_to_pacf(phi: np.ndarray) -> np.ndarray:
    """
    Map AR coefficients phi_1,...,phi_p to partial autocorrelations
    kappa_1,...,kappa_p using the backward Levinson recursion.

    This is the inverse of the *minus-sign* forward recursion used in pacf_to_phi:
        phi_k^(m) = phi_k^(m-1) - kappa_m * phi_{m-k}^(m-1).
    """
    import numpy as np

    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    if p == 0:
        return np.array([], dtype=float)

    phi_m = phi.copy()
    kappa = np.empty(p, dtype=float)

    for m in range(p, 0, -1):
        kappa[m - 1] = phi_m[m - 1]

        if m == 1:
            break

        denom = 1.0 - kappa[m - 1] ** 2
        phi_prev = np.empty(m - 1, dtype=float)

        for k in range(m - 1):
            # Correct inverse for minus-sign forward recursion (NOTE the +):
            phi_prev[k] = (phi_m[k] + kappa[m - 1] * phi_m[m - 2 - k]) / denom

        phi_m = phi_prev

    return kappa



def pacf_to_phi(kappa: np.ndarray) -> np.ndarray:
    """
    Map partial autocorrelations kappa_1,...,kappa_p to AR coefficients
    phi_1,...,phi_p using the forward Levinson recursion.

    Parameters
    ----------
    kappa : array_like, shape (p,)
        Partial autocorrelations / reflection coefficients.

    Returns
    -------
    phi : ndarray, shape (p,)
        AR(p) coefficients.
    """
    kappa = np.asarray(kappa, dtype=float)
    p = len(kappa)
    if p == 0:
        return np.array([], dtype=float)

    phi_prev = np.array([], dtype=float)

    for m in range(1, p + 1):
        phi_m = np.empty(m, dtype=float)
        phi_m[m - 1] = kappa[m - 1]

        for k in range(m - 1):
            # forward recursion:
            # phi_k^(m) = phi_k^(m-1) + kappa_m * phi_{m-k}^(m-1)
            phi_m[k] = phi_prev[k] - kappa[m - 1] * phi_prev[m - 2 - k]

        phi_prev = phi_m

    return phi_prev


import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal

IC = Literal["aic", "bic", "hqic"]

@dataclass
class ARFit:
    p: int
    phi: Optional[np.ndarray]     # shape (p,) or None if p=0
    sigma2: float                 # residual variance
    n_eff: int                    # number of used observations
    ic_values: dict               # {"aic":..., "bic":..., "hqic":...}

@dataclass
class ARDiagnostics:
    p: int
    phi: Optional[np.ndarray]
    eigenvalues: Optional[np.ndarray]
    spectral_radius: float
    stationary_by_companion: bool
    charpoly_roots: Optional[np.ndarray]
    stationary_by_roots: bool
    ic_used: str

def _lagged_matrix(y: np.ndarray, p: int):
    """
    Create (y_p, X_p) for AR(p): y_t = sum_{i=1}^p phi_i y_{t-i} + e_t
    Returns y_p (n_eff,), X_p (n_eff, p)
    """
    if p == 0:
        return y.copy(), np.empty((y.shape[0], 0))
    T = len(y)
    y_p = y[p:].copy()
    X_p = np.column_stack([y[p - i:T - i] for i in range(1, p + 1)])
    return y_p, X_p

def _ols(y: np.ndarray, X: np.ndarray):
    """
    OLS solution: beta, residuals, sigma2, n_eff
    If X has zero columns (p=0), beta is empty and residuals = y.
    """
    n_eff = len(y)
    if X.size == 0:
        resid = y
        sigma2 = float(np.dot(resid, resid) / n_eff)
        return np.empty((0,)), resid, sigma2, n_eff
    # Solve (X'X) beta = X'y with robust numerics
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(np.dot(resid, resid) / n_eff)  # ML variance (not n_eff-p)
    return beta, resid, sigma2, n_eff

def _gaussian_ic(sigma2: float, n_eff: int, k: int):
    """
    Gaussian loglik (up to additive constant across models):
      logL = -n_eff/2*(log(2π) + 1) - n_eff/2 * log(sigma2)
    AIC = -2 logL + 2k; BIC = -2 logL + k log(n_eff); HQIC = -2 logL + 2k log(log n_eff)
    k = number of free parameters (here, p AR coefficients; we de-mean instead of fitting intercept)
    """
    if sigma2 <= 0:
        sigma2 = 1e-300
    logL = -0.5 * n_eff * (np.log(2 * np.pi) + 1.0 + np.log(sigma2))
    aic = -2.0 * logL + 2.0 * k
    bic = -2.0 * logL + k * np.log(max(n_eff, 2))
    hq = -2.0 * logL + 2.0 * k * np.log(np.log(max(n_eff, 3)))
    return {"aic": float(aic), "bic": float(bic), "hqic": float(hq)}

def _companion_and_eigs(phi: np.ndarray):
    """
    Build companion matrix and compute eigenvalues.
    AR(p): y_t = φ1 y_{t-1} + ... + φp y_{t-p} + e_t
    Companion matrix Φ has top row [φ1,...,φp], subdiagonal identity.
    """
    p = len(phi)
    if p == 0:
        return None, np.array([])  # no eigenvalues
    Phi = np.zeros((p, p), dtype=float)
    Phi[0, :] = phi
    if p > 1:
        Phi[1:, :-1] = np.eye(p - 1)
    eigs = np.linalg.eigvals(Phi)
    return Phi, eigs

def _charpoly_roots(phi: np.ndarray):
    """
    Roots of: 1 - φ1 z - φ2 z^2 - ... - φp z^p = 0
    Coeffs (descending powers): [-φ_p, ..., -φ_1, 1]
    """
    p = len(phi)
    if p == 0:
        return np.array([])
    coefs = np.r_[ -phi[::-1], 1.0 ]
    return np.roots(coefs)

def select_ar_and_check_stationarity(
    y,
    max_p: int = 12,
    ic: IC = "bic",
    demean: bool = True,
    eps: float = 1e-12,
) -> ARDiagnostics:
    """
    Pure-NumPy AR order selection + stationarity check.

    Parameters
    ----------
    y : array-like (1D)
        Time series.
    max_p : int
        Consider p in {0,...,max_p}.
    ic : {"aic","bic","hqic"}
        Information criterion for order selection.
    demean : bool
        If True, subtract sample mean before fitting (no intercept in regression).
    eps : float
        Tolerance for strict stationarity inequalities.

    Returns
    -------
    ARDiagnostics
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if y.size < 5:
        raise ValueError("Need at least 5 observations after dropping NaNs.")
    y_fit = y - y.mean() if demean else y.copy()

    fits: list[ARFit] = []
    T = len(y_fit)
    max_p = min(max_p, T - 2)  # ensure at least some rows in design matrix

    for p in range(max_p + 1):
        y_p, X_p = _lagged_matrix(y_fit, p)
        phi, resid, sigma2, n_eff = _ols(y_p, X_p)
        ic_vals = _gaussian_ic(sigma2, n_eff, k=p)
        fits.append(ARFit(p=p, phi=None if p == 0 else phi, sigma2=sigma2, n_eff=n_eff, ic_values=ic_vals))

    # pick best p by chosen IC
    ic_name = ic.lower()
    best = min(fits, key=lambda f: f.ic_values[ic_name])

    # stationarity checks
    if best.p == 0:
        return ARDiagnostics(
            p=0,
            phi=None,
            eigenvalues=None,
            spectral_radius=0.0,
            stationary_by_companion=True,   # white noise is stationary
            charpoly_roots=None,
            stationary_by_roots=True,
            ic_used=ic_name,
        )

    phi = best.phi
    _, eigs = _companion_and_eigs(phi)
    spectral_radius = float(np.max(np.abs(eigs)))
    stationary_companion = bool(spectral_radius < 1.0 - eps)

    roots = _charpoly_roots(phi)
    stationary_roots = bool(np.all(np.abs(roots) > 1.0 + eps))

    return ARDiagnostics(
        p=best.p,
        phi=phi,
        eigenvalues=eigs,
        spectral_radius=spectral_radius,
        stationary_by_companion=stationary_companion,
        charpoly_roots=roots,
        stationary_by_roots=stationary_roots,
        ic_used=ic_name,
    )
