"""Example script demonstrating how to use the Precision MMM package (updated API)."""

from __future__ import annotations

import numpy as np

from precision import (
    Priors,
    build_hierarchy,
    compute_contributions_from_params,
    make_target_log_prob_fn,
    posterior_mean,
    run_nuts,
    summarise_decay_rates,
)


def main(seed: int = 123) -> None:
    rng = np.random.default_rng(seed)

    # --- 1) Define hierarchy -------------------------------------------------
    levels = ["tactical", "platform", "channel"]
    spec = {
        "channel100": {
            "platform110": ["tactical111", "tactical112", "tactical113"],
            "platform120": ["tactical121", "tactical122", "tactical123"],
            "platform130": ["tactical131", "tactical132", "tactical133"],
        },
        "channel200": {
            "platform210": ["tactical211", "tactical212", "tactical213"],
            "platform220": ["tactical221", "tactical222", "tactical223"],
            "platform230": ["tactical231", "tactical232", "tactical233"],
        },
        "channel300": {
            "platform310": ["tactical311", "tactical312", "tactical313"],
            "platform320": ["tactical321", "tactical322", "tactical323"],
            "platform330": ["tactical331", "tactical332", "tactical333"],
        },
    }
    hierarchy = build_hierarchy(spec, levels)
    P = hierarchy.num_platforms
    T = 52
    N_t = hierarchy.num_tacticals

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
    platform_series = tactical_series @ hierarchy.M_tp

    # Platform-level effects (positive, log-lift in the model)
    beta_channel_base = np.array([0.50, 0.30, 0.20])
    beta_true_platform = beta_channel_base[hierarchy.p_to_c] + rng.normal(scale=0.05, size=P)
    beta_true_platform = np.clip(beta_true_platform, 0.02, None)

    # Controls & outcome
    gamma_true = rng.normal(scale=0.2, size=Z_controls.shape[1])
    beta0_true = 1.0
    mean = beta0_true + platform_series @ beta_true_platform + Z_controls @ gamma_true
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
