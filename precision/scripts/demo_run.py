"""Example script demonstrating how to use the Precision MMM package (updated API)."""

from __future__ import annotations

import numpy as np

from precision import (
    Priors,
    compute_contributions_from_params,
    make_target_log_prob_fn,
    posterior_mean,
    run_nuts,
    summarise_decay_rates,
)
from precision.precision.hierarchy import pad_ragged_tree, build_hierarchy
from precision.precision.summaries import compute_contribution_arrays


def main(seed: int = 123) -> None:
    rng = np.random.default_rng(seed)

    # --- 1) Define hierarchy -------------------------------------------------
    levels = ["level1", "level2", "level3"]  # bottom -> top
    tree_ragged = {
        "L3_A": {"L2_X": ["L1_a", "L1_b"]},
        "L3_B": ["L1_c", "L1_d"],
    }
    tree_uniform = pad_ragged_tree(tree_ragged, levels)
    hierarchy = build_hierarchy(tree_uniform, levels)

    leaf_level = hierarchy.levels[0]
    beta_level = hierarchy.levels[1]
    top_level = hierarchy.levels[2]

    T = 52
    N_t = hierarchy.size(leaf_level)
    N_beta = hierarchy.size(beta_level)
    N_top = hierarchy.size(top_level)

    # --- 2) Simulate data (consistent with platform-hier structure) ----------
    U_tactical = rng.gamma(shape=2.0, scale=1.0, size=(T, N_t))
    Z_controls = rng.normal(size=(T, 2))
    control_names = [f"control_{j}" for j in range(Z_controls.shape[1])]

    # True decay (for simulation only)
    true_delta = rng.beta(a=2.0, b=5.0, size=N_t)

    # Adstock with normalization to match modeling convention
    def adstock_geometric_np(U: np.ndarray, d: np.ndarray, normalize: bool = True) -> np.ndarray:
        A = np.zeros_like(U, dtype=float)
        for t in range(U.shape[0]):
            A[t] = U[t] if t == 0 else U[t] + d * A[t - 1]
        return A * (1.0 - d) if normalize else A

    tactical_series = adstock_geometric_np(U_tactical, true_delta, normalize=True)
    mid_series = tactical_series @ hierarchy.map(leaf_level, beta_level)

    # Level-2 effects (positive, log-lift in the model)
    beta_top_base = np.linspace(0.5, 0.3, N_top)
    idx_mid_to_top = hierarchy.index_map(beta_level, top_level)
    beta_true_mid = beta_top_base[idx_mid_to_top] + rng.normal(scale=0.05, size=N_beta)
    beta_true_mid = np.clip(beta_true_mid, 0.02, None)

    # Controls & outcome
    gamma_true = rng.normal(scale=0.2, size=Z_controls.shape[1])
    beta0_true = 1.0
    mean = beta0_true + mid_series @ beta_true_mid + Z_controls @ gamma_true
    y = mean + rng.normal(scale=1.0, size=T)

    # --- 3) Build target log-prob (uses scaling + centering by default) ------
    priors = Priors(
        # defaults: decay_mode="half_life", beta_structure="platform_hier",
        # standardize_media="pre_adstock_tactical", center_y=True
        # tweak here if you want: e.g., use_saturation=True, sparsity_prior="horseshoe", etc.
    )
    target_fn, dims, param_spec = make_target_log_prob_fn(
        y=y,
        U_tactical=U_tactical,
        Z_controls=Z_controls,
        hierarchy=hierarchy,
        priors=priors,
        normalize_adstock=True,
    )

    # --- 4) Sample with NUTS -------------------------------------------------
    samples = run_nuts(
        target_fn,
        dims,
        param_spec,
        num_chains=2,
        num_burnin=250,
        num_samples=250,
        init_step_size=0.15,
        seed=seed,
    )

    # --- 5) Quick posterior summaries ---------------------------------------
    post = posterior_mean(samples)
    decay_summary = summarise_decay_rates(samples, hierarchy)
    print(decay_summary.head())

    leaf_level_resolved, beta_level_resolved, pool_parent = priors.resolve_beta_levels(
        hierarchy.levels
    )
    beta_by_level = post.get("beta_by_level", {})
    beta_est = beta_by_level.get(beta_level_resolved)
    if beta_est is None:
        raise ValueError(f"Posterior mean for beta level {beta_level_resolved!r} not available")
    print(
        "Resolved beta placement:",
        f"leaf={leaf_level_resolved}",
        f"beta={beta_level_resolved}",
        f"pool_parent={pool_parent}",
    )

    contrib_arrays = compute_contribution_arrays(
        U_raw=U_tactical,
        delta=post["delta"],
        beta=beta_est,
        beta_level=beta_level_resolved,
        hierarchy=hierarchy,
        leaf_level=leaf_level_resolved,
        normalize_adstock=True,
        use_saturation=priors.use_saturation,
        s_sat=post.get("s_sat"),
        tactical_rescale=np.asarray(dims.get("tactical_rescale", []), dtype=float)
        if "tactical_rescale" in dims
        else None,
        levels=[lvl for lvl in hierarchy.levels if lvl != leaf_level_resolved],
    )
    for level_name, values in contrib_arrays.items():
        print(f"Contribution array {level_name}: shape={values.shape}")

    # --- 6) Contributions on ORIGINAL scale ---------------------------------
    # Note: the model standardizes media and centers y.
    #       Use dims["tactical_rescale"] to re-scale betas
    #       and dims["y_mean"] to offset the intercept back to data scale.
    contrib = compute_contributions_from_params(
        y=y,
        U_tactical=U_tactical,
        Z_controls=Z_controls,
        control_names=control_names,
        hierarchy=hierarchy,
        beta0=post["beta0"],
        beta_channel=post.get("beta_channel"),
        beta_platform=post.get("beta_platform"),
        beta_tactical=post.get("beta_tactical"),
        gamma=post.get("gamma"),
        delta=post["delta"],
        normalize_adstock=True,
        beta0_offset=float(dims.get("y_mean", 0.0)),
        tactical_scale=np.asarray(dims.get("tactical_rescale", []), dtype=float)
        if "tactical_rescale" in dims
        else None,
    )
    print("Channel totals:\n", contrib.channel_totals)
    print("Platform totals:\n", contrib.platform_totals)
    print("Controls totals:\n", contrib.controls_totals)
    print("Intercept total:\n", contrib.intercept_total)
    print("Fitted total:\n", contrib.fitted_total)


if __name__ == "__main__":
    main()
