"""Test package initialisation for Precision tests."""

import sys
import types


def _ensure_tf_keras_stub() -> None:
    if "tf_keras" in sys.modules:
        return

    tf_keras = types.ModuleType("tf_keras")
    api = types.ModuleType("tf_keras.api")
    v1 = types.ModuleType("tf_keras.api._v1")
    keras = types.ModuleType("tf_keras.api._v1.keras")
    internal = types.ModuleType("tf_keras.api._v1.keras.__internal__")
    utils = types.ModuleType("tf_keras.api._v1.keras.__internal__.utils")
    legacy = types.ModuleType("tf_keras.api._v1.keras.__internal__.legacy")
    layers = types.ModuleType("tf_keras.api._v1.keras.__internal__.legacy.layers")

    # Wire the package hierarchy.
    tf_keras.api = api
    api._v1 = v1
    v1.keras = keras
    keras.__internal__ = internal
    def _register_symbolic_tensor_type(*args, **kwargs):
        return None

    utils.register_symbolic_tensor_type = _register_symbolic_tensor_type
    internal.utils = utils
    internal.legacy = legacy
    activations = types.ModuleType("tf_keras.activations")
    initializers = types.ModuleType("tf_keras.initializers")
    regularizers = types.ModuleType("tf_keras.regularizers")
    utils_public = types.ModuleType("tf_keras.utils")
    class _Sequential:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _Layer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    legacy.layers = layers
    tf_keras.__internal__ = internal
    tf_keras.layers = layers
    tf_keras.activations = activations
    tf_keras.Sequential = _Sequential
    layers.Sequential = _Sequential
    layers.Layer = _Layer

    class _Lambda(_Layer):
        def __init__(self, function=None, *args, **kwargs):
            super().__init__(function, *args, **kwargs)
            self.function = function

        def __call__(self, *args, **kwargs):  # pragma: no cover - stub
            if callable(self.function):
                return self.function(*args, **kwargs)
            return None

    layers.Lambda = _Lambda

    class _Wrapper(_Layer):
        pass

    layers.Wrapper = _Wrapper
    tf_keras.initializers = initializers
    tf_keras.regularizers = regularizers
    tf_keras.utils = utils_public

    class _RandomNormal:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, shape, dtype=None):  # pragma: no cover - simple stub
            return 0.0

    initializers.RandomNormal = _RandomNormal
    initializers.Initializer = type("Initializer", (), {})
    class _ConstantInitializer:
        def __init__(self, value):
            self.value = value

        def __call__(self, shape=None, dtype=None):  # pragma: no cover
            return self.value

    def constant(value):
        return _ConstantInitializer(value)

    initializers.constant = constant
    regularizers.Regularizer = type("Regularizer", (), {})

    sys.modules["tf_keras"] = tf_keras
    sys.modules["tf_keras.api"] = api
    sys.modules["tf_keras.api._v1"] = v1
    sys.modules["tf_keras.api._v1.keras"] = keras
    sys.modules["tf_keras.api._v1.keras.__internal__"] = internal
    sys.modules["tf_keras.api._v1.keras.__internal__.utils"] = utils
    sys.modules["tf_keras.api._v1.keras.__internal__.legacy"] = legacy
    sys.modules["tf_keras.api._v1.keras.__internal__.legacy.layers"] = layers
    sys.modules["tf_keras.activations"] = activations
    sys.modules["tf_keras.initializers"] = initializers
    sys.modules["tf_keras.regularizers"] = regularizers
    _CUSTOM_OBJECTS: dict[str, object] = {}

    def get_custom_objects():  # pragma: no cover
        return _CUSTOM_OBJECTS

    utils_public.get_custom_objects = get_custom_objects
    sys.modules["tf_keras.utils"] = utils_public


_ensure_tf_keras_stub()
