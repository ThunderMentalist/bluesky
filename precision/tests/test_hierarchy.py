import numpy as np
import pytest

from precision.precision.hierarchy import Hierarchy, build_hierarchy


@pytest.fixture()
def sample_spec():
    return {
        "channelA": {"platformA1": ["t1", "t2"], "platformA2": ["t3"]},
        "channelB": {"platformB1": ["t4"]},
    }


def test_build_hierarchy_shapes(sample_spec):
    hierarchy = build_hierarchy(sample_spec)

    assert hierarchy.M_tp.shape == (4, 3)
    assert hierarchy.M_tc.shape == (4, 2)
    np.testing.assert_allclose(hierarchy.M_tp.sum(axis=1), 1.0)
    np.testing.assert_allclose(hierarchy.M_tc.sum(axis=1), 1.0)


def test_build_hierarchy_sorted(sample_spec):
    hierarchy = build_hierarchy(sample_spec, keep_order=False)

    assert hierarchy.channel_names == sorted(sample_spec.keys())
    assert hierarchy.platform_names == sorted({p for v in sample_spec.values() for p in v.keys()})


def test_build_hierarchy_duplicate_detection(sample_spec):
    spec = {"channelA": {"platformA": ["t1", "t1"]}}
    with pytest.raises(ValueError):
        build_hierarchy(spec)


def test_hierarchy_properties(sample_spec):
    hierarchy = build_hierarchy(sample_spec)
    assert hierarchy.num_channels == 2
    assert hierarchy.num_platforms == 3
    assert hierarchy.num_tacticals == 4
