import numpy as np

from precision.precision.scaling import (
    apply_pre_adstock_tactical,
    center_y,
    fit_media_scales_pre_adstock_tactical,
)


def test_fit_media_scales_stat_modes():
    U = np.array([[1.0, 3.0], [3.0, 9.0], [5.0, 27.0]])

    median = fit_media_scales_pre_adstock_tactical(U, stat="median")
    mean = fit_media_scales_pre_adstock_tactical(U, stat="mean")
    p95 = fit_media_scales_pre_adstock_tactical(U, stat="p95")

    np.testing.assert_allclose(median, [3.0, 9.0])
    np.testing.assert_allclose(mean, [3.0, 13.0])
    assert np.all(p95 >= median)


def test_apply_pre_adstock_tactical_validates_shape():
    U = np.ones((2, 3))
    s = np.array([1.0, 2.0, 4.0])
    scaled = apply_pre_adstock_tactical(U, s)
    np.testing.assert_allclose(scaled, np.array([[1.0, 0.5, 0.25], [1.0, 0.5, 0.25]]))


def test_center_y_returns_mean():
    y = np.array([1.0, 3.0, 5.0])
    centred, mean = center_y(y)
    np.testing.assert_allclose(centred, [-2.0, 0.0, 2.0])
    assert mean == 3.0
