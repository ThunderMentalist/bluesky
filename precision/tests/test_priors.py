import dataclasses
import pytest

from precision.precision.priors import Priors


def test_priors_defaults_are_reasonable():
    priors = Priors()
    assert priors.beta0_sd == 5.0
    assert priors.decay_mode == "half_life"
    assert priors.standardize_media == "pre_adstock_tactical"


def test_priors_are_frozen_dataclass():
    priors = Priors()
    with pytest.raises(dataclasses.FrozenInstanceError):
        priors.beta0_sd = 10.0  # type: ignore[misc]
