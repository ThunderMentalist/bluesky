import numpy as np
import pytest

from precision.precision.hierarchy import Hierarchy, build_hierarchy, pad_ragged_tree


@pytest.fixture()
def sample_tree():
    return {
        "channelA": {"platformA1": ["t1", "t2"], "platformA2": ["t3"]},
        "channelB": {"platformB1": ["t4"]},
    }


@pytest.fixture()
def levels() -> list[str]:
    return ["tactical", "platform", "channel"]


def test_build_hierarchy_shapes(sample_tree, levels):
    hierarchy = build_hierarchy(sample_tree, levels)

    assert hierarchy.M_tp.shape == (4, 3)
    assert hierarchy.M_tc.shape == (4, 2)
    np.testing.assert_allclose(hierarchy.M_tp.sum(axis=1), 1.0)
    np.testing.assert_allclose(hierarchy.M_tc.sum(axis=1), 1.0)


def test_build_hierarchy_names(sample_tree, levels):
    hierarchy = build_hierarchy(sample_tree, levels)

    assert hierarchy.names["channel"] == list(sample_tree.keys())
    assert hierarchy.names["platform"] == [
        "platformA1",
        "platformA2",
        "platformB1",
    ]


def test_build_hierarchy_duplicate_detection(levels):
    spec = {"channelA": {"platformA": ["t1", "t1"]}}
    with pytest.raises(ValueError):
        build_hierarchy(spec, levels)


def test_hierarchy_properties(sample_tree, levels):
    hierarchy = build_hierarchy(sample_tree, levels)
    assert hierarchy.num_channels == 2
    assert hierarchy.num_platforms == 3
    assert hierarchy.num_tacticals == 4


def test_map_products(sample_tree, levels):
    hierarchy = build_hierarchy(sample_tree, levels)

    np.testing.assert_allclose(
        hierarchy.map("tactical", "channel"),
        hierarchy.M_tc,
    )

    np.testing.assert_allclose(
        hierarchy.map("tactical", "platform"),
        hierarchy.M_tp,
    )

    np.testing.assert_array_equal(
        hierarchy.index_map("platform", "channel"),
        hierarchy.p_to_c,
    )

    np.testing.assert_array_equal(
        hierarchy.index_map("tactical", "platform"),
        hierarchy.t_to_p,
    )


def test_pad_ragged_tree_is_noop_for_uniform(sample_tree, levels):
    padded = pad_ragged_tree(sample_tree, levels)
    assert padded == sample_tree


def test_pad_ragged_tree_fills_missing_levels(levels):
    ragged = {
        "channelA": ["t1", "t2"],
        "channelB": {"platformB1": ["t3"]},
    }

    padded = pad_ragged_tree(ragged, levels)

    placeholder = "__auto__:platform:channelA"
    assert padded["channelA"] == {placeholder: ["t1", "t2"]}
    assert padded["channelB"] == ragged["channelB"]

    hierarchy = build_hierarchy(padded, levels)

    assert placeholder in hierarchy.names["platform"]
    assert "t1" in hierarchy.names["tactical"]
