import numpy as np
import tensorflow as tf

from precision.precision.adstock import adstock_geometric_np, adstock_geometric_tf


def test_adstock_geometric_np_matches_manual():
    u = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    delta = np.array([0.5, 0.25])

    expected = np.array(
        [
            [1.0, 0.0],
            [0.5, 2.0],
            [3.25, 0.5],
        ]
    )

    out = adstock_geometric_np(u, delta)
    np.testing.assert_allclose(out, expected)


def test_adstock_geometric_np_normalized():
    u = np.ones((4, 1))
    delta = np.array([0.5])

    out = adstock_geometric_np(u, delta, normalize=True)
    np.testing.assert_allclose(out[:, 0], [0.5, 0.75, 0.875, 0.9375])


def test_adstock_geometric_tf_matches_numpy():
    u = tf.constant([[1.0], [2.0], [0.0]], dtype=tf.float64)
    delta = tf.constant([0.2], dtype=tf.float64)

    tf_out = adstock_geometric_tf(u, delta)
    np_out = adstock_geometric_np(u.numpy(), delta.numpy())

    np.testing.assert_allclose(tf_out.numpy(), np_out)
