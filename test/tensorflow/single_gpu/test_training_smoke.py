"""Smoke test: train a small Keras model, verify loss decreases, checkpoint round-trip."""

import os
import tempfile

import pytest
import tensorflow as tf

pytestmark = pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="GPU only")


def _build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(8, 8)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )


def test_training_loss_decreases():
    tf.keras.utils.set_random_seed(42)
    model = _build_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    x = tf.random.normal([32, 8, 8])
    y = tf.random.uniform([32], maxval=10, dtype=tf.int32)

    losses = []
    for _ in range(10):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(loss))

    assert losses[-1] < 0.9 * losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_checkpoint_roundtrip():
    tf.keras.utils.set_random_seed(7)
    model = _build_model()
    optimizer = tf.keras.optimizers.Adam()
    # Build optimizer state by running one step
    x = tf.random.normal([4, 8, 8])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(model(x, training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    sample = tf.random.normal([2, 8, 8])
    original = model(sample, training=False)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "ckpt")
        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt.write(prefix)

        restored_model = _build_model()
        restored_opt = tf.keras.optimizers.Adam()
        # Build restored model and optimizer state to match shapes
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(restored_model(x, training=True))
        grads = tape.gradient(loss, restored_model.trainable_variables)
        restored_opt.apply_gradients(zip(grads, restored_model.trainable_variables))

        tf.train.Checkpoint(model=restored_model, optimizer=restored_opt).read(
            prefix
        ).assert_consumed()
        restored = restored_model(sample, training=False)

    max_diff = float(tf.reduce_max(tf.abs(original - restored)))
    assert max_diff < 1e-5, f"checkpoint round-trip diverged by {max_diff}"
