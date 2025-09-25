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
