import numpy as np
import tensorflow as tf

from precision.precision.hierarchy import build_hierarchy
from precision.precision.posterior import ParamSpec, _saturate_log1p, make_target_log_prob_fn
from precision.precision.priors import Priors


def _make_simple_hierarchy():
    return build_hierarchy({"c1": {"p1": ["t1"]}}, ["tactical", "platform", "channel"])


def test_saturate_log1p_behaviour():
    x = tf.constant([0.0, 1.0, 2.0], dtype=tf.float64)
    s = tf.constant([0.5], dtype=tf.float64)
    out = _saturate_log1p(x, s)
    assert out.shape == (3,)
    assert float(out[-1]) < 2.0  # saturation reduces magnitude


def test_make_target_log_prob_fn_runs():
    hierarchy = _make_simple_hierarchy()
    y = np.array([1.0, 2.0], dtype=float)
    U = np.array([[1.0], [0.5]], dtype=float)
    Z = np.array([[1.0], [1.0]], dtype=float)

    priors = Priors(center_y=False, standardize_media="none")
    target_fn, dims, spec = make_target_log_prob_fn(
        y=y,
        U_tactical=U,
        Z_controls=Z,
        hierarchy=hierarchy,
        normalize_adstock=False,
        priors=priors,
    )

    assert isinstance(dims, dict)
    assert any(isinstance(s, ParamSpec) for s in spec)

    # Evaluate log prob at parameter inits to ensure finiteness.
    args = []
    for s in spec:
        init_tensor = tf.convert_to_tensor(s.init, dtype=tf.float64)
        if s.shape:
            args.append(tf.broadcast_to(init_tensor, (1, *s.shape)))
        else:
            args.append(tf.reshape(init_tensor, (1,)))
    value = target_fn(*args)
    assert tf.math.is_finite(value).numpy()
