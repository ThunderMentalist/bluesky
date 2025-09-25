"""Example script demonstrating how to use the Precision MMM package."""

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
    hierarchy = build_hierarchy(spec)

    T = 52
    num_tacticals = hierarchy.num_tacticals
    num_controls = 2

    U_tactical = rng.gamma(shape=2.0, scale=1.0, size=(T, num_tacticals))
    Z_controls = rng.normal(size=(T, num_controls))
    control_names = [f"control_{idx}" for idx in range(num_controls)]

    true_delta = rng.beta(a=2.0, b=5.0, size=num_tacticals)
    adstocked = compute_adstock_np(U_tactical, true_delta)
    beta_true = np.array([0.5, 0.3, 0.2])
    beta_tactical = hierarchy.M_tc @ beta_true
    channel_signal = (adstocked * beta_tactical[None, :]) @ hierarchy.M_tc
    gamma_true = rng.normal(scale=0.2, size=num_controls)
    mean = 1.0 + channel_signal.sum(axis=1) + Z_controls @ gamma_true
    y = mean + rng.normal(scale=1.0, size=T)

    target_fn, dims = make_target_log_prob_fn(
        y=y,
        U_tactical=U_tactical,
        Z_controls=Z_controls,
        hierarchy=hierarchy,
        priors=Priors(),
    )

    samples = run_nuts(
        target_fn,
        dims,
        num_chains=2,
        num_burnin=200,
        num_samples=200,
        init_step_size=0.15,
        seed=seed,
    )

    post = posterior_mean(samples)
    decay_summary = summarise_decay_rates(samples, hierarchy)
    print(decay_summary.head())

    contrib = compute_contributions_from_params(
        y=y,
        U_tactical=U_tactical,
        Z_controls=Z_controls,
        control_names=control_names,
        hierarchy=hierarchy,
        beta0=post["beta0"],
        beta_channel=post["beta_channel"],
        gamma=post["gamma"],
        delta=post["delta"],
    )
    print("Channel totals:\n", contrib.channel_totals)


def compute_adstock_np(U_tactical: np.ndarray, delta: np.ndarray) -> np.ndarray:
    from precision.adstock import adstock_geometric_np

    return adstock_geometric_np(U_tactical, delta)


if __name__ == "__main__":
    main()
