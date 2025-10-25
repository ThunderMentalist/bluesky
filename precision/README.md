# Precision MMM

Precision is a TensorFlow Probability based Bayesian marketing mix modelling (MMM) toolkit
that implements a channel → platform → tactical hierarchy with geometric adstock and
per-tactical decay priors. The package mirrors the specification provided in the project
brief and exposes helpers for sampling, posterior summarisation, and contribution
analysis.

## Project layout

```
precision/
├── precision/          # Python package with reusable modules
├── scripts/            # Command line utilities (placeholders)
├── tests/              # Pytest test-suite
└── README.md           # This file
```

Key modules inside the `precision` package:

- `hierarchy.py`: constructs hierarchy matrices and stores metadata.
- `adstock.py`: TensorFlow and NumPy implementations of geometric adstock.
- `posterior.py`: builds the target log-posterior function.
- `sampling.py`: runs the No-U-Turn Sampler (NUTS) and stores samples.
- `summaries.py`: posterior summaries and contribution accounting utilities.

## Getting started

Install dependencies (TensorFlow, TensorFlow Probability, NumPy, pandas, pytest) using
`pip install -r requirements.txt` once provided or manually. Then run the tests:

```
pytest
```

## Example usage

See `scripts/demo_run.py` for an end-to-end example that simulates data, runs NUTS, and
prints summarised decay rates and channel contributions.

### PSIS-LOO stacking and metric safeguards

```python
from precision.ensemble import ensemble
from precision.hierarchy import build_hierarchy
import numpy as np

# Example hierarchy
levels = ["tactical", "platform", "channel"]
hierarchy = build_hierarchy({
    "channel_TV": {"platform_TV": ["tv_spot_a", "tv_spot_b"]},
    "channel_Search": {"platform_Search": ["search_brand", "search_generic"]},
}, levels)

T = 100  # time periods
N = hierarchy.num_tacticals

U_metrics = {
    "impressions": np.random.gamma(5.0, 20.0, size=(T, N)),
    "clicks": np.random.gamma(3.0, 2.0, size=(T, N)),
    "conversions": np.random.gamma(2.0, 0.5, size=(T, N)),
}

y = np.random.normal(1000.0, 50.0, size=T)

result = ensemble(
    hierarchy=hierarchy,
    y=y,
    U_metrics=U_metrics,
    metric_lags={"clicks": 1, "conversions": 2},
    offline_channels=["channel_TV"],
    weight_scheme="psis_loo",  # default stacking scheme using PSIS-LOO
)

print(result.weights)
```

The helper applies per-metric lags, performs leakage checks, excludes offline tacticals
from downstream metrics, and stacks models using PSIS-LOO log predictive densities.
