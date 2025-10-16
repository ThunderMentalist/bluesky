"""Adstock operators for the Precision MMM package."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

DTYPE = tf.float64


def adstock_geometric_tf(
    U_tactical: tf.Tensor,
    delta: tf.Tensor,
    *,
    normalize: bool = False,
) -> tf.Tensor:
    """Apply geometric adstock in TensorFlow.

    Args:
        U_tactical: Tensor of shape ``[T, N_t]`` representing tactical inputs.
        delta: Tensor of shape ``[N_t]`` with decay factors in ``(0, 1)``.
        normalize: If ``True``, scales the cumulative signal by ``(1 - delta)``.

    Returns:
        Tensor with the adstocked tacticals, shape ``[T, N_t]``.
    """

    U_tactical = tf.convert_to_tensor(U_tactical, dtype=DTYPE)
    delta = tf.reshape(tf.convert_to_tensor(delta, dtype=DTYPE), [-1])

    def step(previous: tf.Tensor, current: tf.Tensor) -> tf.Tensor:
        return current + delta * previous

    adstocked = tf.scan(step, U_tactical, initializer=tf.zeros_like(U_tactical[0]))
    if normalize:
        adstocked = adstocked * tf.reshape(1.0 - delta, [1, -1])
    return adstocked


def adstock_geometric_np(
    U_tactical: np.ndarray,
    delta: np.ndarray,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Apply geometric adstock in NumPy."""

    U = np.asarray(U_tactical, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    adstocked = np.zeros_like(U)
    for t in range(U.shape[0]):
        if t == 0:
            adstocked[t] = U[t]
        else:
            adstocked[t] = U[t] + d * adstocked[t - 1]
    if normalize:
        adstocked = adstocked * (1.0 - d)
    return adstocked


__all__ = ["adstock_geometric_np", "adstock_geometric_tf", "DTYPE"]
