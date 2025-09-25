import importlib.util
import pytest


def test_build_hierarchy_shapes():
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy is required for hierarchy tests")

    import numpy as np  # type: ignore[import-untyped]
    from precision.hierarchy import build_hierarchy

    spec = {
        "channel100": {
            "platform110": ["tactical111", "tactical112"],
            "platform120": ["tactical121"],
        },
        "channel200": {
            "platform210": ["tactical211"],
        },
    }

    hierarchy = build_hierarchy(spec)

    assert hierarchy.M_tp.shape == (4, 3)
    assert hierarchy.M_tc.shape == (4, 2)
    assert np.allclose(hierarchy.M_tp.sum(axis=1), 1.0)
    assert np.allclose(hierarchy.M_tc.sum(axis=1), 1.0)
