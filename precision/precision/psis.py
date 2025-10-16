from __future__ import annotations

import numpy as np

EPS = 1e-12


def _logsumexp(a: np.ndarray, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.clip(np.sum(np.exp(a - m), axis=axis, keepdims=True), EPS, np.inf))
    return s if keepdims else np.squeeze(s, axis=axis)


def _psis_smooth_log_weights(log_raw_w: np.ndarray, tail_fraction: float = 0.2) -> np.ndarray:
    """
    Pareto-smooth the largest raw importance weights (Vehtari et al., 2017).
    `log_raw_w`: [S] log of raw importance ratios (unnormalized).
    Returns log of smoothed, normalized importance weights.
    This is a compact, dependency-free implementation:
      1) normalize raw w
      2) pick largest M = ceil(tail_fraction*S)
      3) fit GPD to tail in log space (method of moments)
      4) replace tail with expected order stats from fitted GPD
    """

    S = log_raw_w.shape[0]
    lw = log_raw_w - _logsumexp(log_raw_w)  # normalize
    w = np.exp(lw)
    # tail selection
    M = max(1, int(np.ceil(tail_fraction * S)))
    idx = np.argsort(w)
    tail_idx = idx[-M:]
    tail = w[tail_idx]
    # Transform to exceedances over min tail
    u = tail.min()
    z = (tail - u) / max(EPS, (tail.max() - u))
    z = np.clip(z, EPS, 1.0 - EPS)
    # Method-of-moments GPD (approx)
    m1 = z.mean()
    m2 = ((z - m1) ** 2).mean()
    k = 0.5 * (1 - (m1**2) / max(EPS, m2))  # shape; crude but stable for stacking use
    k = np.clip(k, -0.5, 0.9)  # keep k in a sane range
    # Expected order stats for tail ranks
    ranks = np.arange(1, M + 1)
    # smooth tail weights ~ Beta order stats heuristic
    smooth = (ranks / (M + 1.0)) ** (-k)  # decreasing if k>0
    smooth = smooth / smooth.sum() * tail.sum()
    w_smooth = w.copy()
    w_smooth[tail_idx] = smooth
    lw_smooth = np.log(np.clip(w_smooth, EPS, np.inf))
    lw_smooth = lw_smooth - _logsumexp(lw_smooth)
    return lw_smooth


def psis_loo_pointwise(loglik_draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PSIS-LOO pointwise log predictive density for a single model.
    Args:
      loglik_draws: [S, T] log p(y_t | theta_s) per draw s and time t.
    Returns:
      (loo_lpd: [T], pareto_k: [T]) approximate LOO log predictive density for each t,
      and a simple k-diagnostic (here we return a proxy; true k needs full GPD fit per t).
    """

    S, T = loglik_draws.shape
    loo_lpd = np.zeros(T)
    pareto_k = np.zeros(T)
    for t in range(T):
        ll = loglik_draws[:, t]
        # raw importance ratios r_s ∝ 1 / p(y_t | theta_s) => log r_s = -ll + const
        log_r = -ll
        lw = _psis_smooth_log_weights(log_r)  # log smoothed & normalized weights
        # loo predictive density: log( sum_s w_s * p(y_t|theta_s) )
        loo_lpd[t] = _logsumexp(lw + ll)
        # k proxy: tail heaviness inferred from smoothing step (crude)
        pareto_k[t] = np.clip(np.std(np.exp(lw)) * 0.5, 0.0, 1.0)
    return loo_lpd, pareto_k


def stacking_weights_from_pointwise_logdens(
    logdens_by_model: dict[str, np.ndarray],
    *,
    lr: float = 0.1,
    max_iter: int = 4000,
    tol: float = 1e-7,
) -> dict[str, float]:
    """
    Given per-model pointwise log densities (e.g., LOO lpd per t), find weights w that maximize
    sum_t log( sum_m w_m * exp(logdens_m[t]) ), w_m >= 0, sum w = 1.
    Uses exponentiated-gradient updates.
    """

    keys = list(logdens_by_model.keys())
    if not keys:
        return {}
    T = min(v.shape[0] for v in logdens_by_model.values())
    L = np.stack([logdens_by_model[k][:T] for k in keys], axis=0)  # [M, T]
    M = L.shape[0]
    w = np.ones(M, dtype=float) / M
    prev = -np.inf
    for _ in range(max_iter):
        # denom_t = sum_m w_m * exp(L_m,t)
        maxL = np.max(L, axis=0, keepdims=True)
        scaled = np.exp(L - maxL)
        denom = np.dot(w, scaled)  # [T]
        denom = np.clip(denom, EPS, np.inf)
        obj = float(np.sum(np.log(denom) + np.squeeze(maxL, 0)))
        # gradient wrt w_m: sum_t exp(L_m,t - maxL_t) / denom_t
        grad = np.sum(scaled / denom, axis=1)  # [M]
        w_new = w * np.exp(lr * grad)
        w_new = np.clip(w_new, 1e-16, np.inf)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < tol:
            w = w_new
            prev = obj
            break
        w, prev = w_new, obj
    return {keys[i]: float(w[i]) for i in range(len(keys))}
