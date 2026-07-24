"""Mixed-precision (float16) forward/backward smoke tests on GPU."""

import pytest
import tensorflow as tf

pytestmark = pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"),
    reason="mixed_float16 requires GPU for actual fp16 ops",
)


def _build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )


def test_mixed_precision_forward_backward():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = _build_model()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        x = tf.random.normal([16, 32])
        y = tf.random.uniform([16], maxval=10, dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        assert grads, "no gradients produced"
        for g in grads:
            assert g is not None, "gradient was None"
            assert bool(tf.reduce_all(tf.math.is_finite(g))), "non-finite values in gradient"
    finally:
        tf.keras.mixed_precision.set_global_policy("float32")


def test_mixed_precision_dtype_mixing():
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = _build_model()
        x = tf.random.normal([4, 32])
        out = model(x, training=False)

        assert out.dtype == tf.float16, f"expected float16 output, got {out.dtype}"
        for var in model.trainable_variables:
            assert var.dtype == tf.float32, f"expected float32 variable {var.name}, got {var.dtype}"
    finally:
        tf.keras.mixed_precision.set_global_policy("float32")
