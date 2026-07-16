"""Smoke test: train a Conv2D model via model.fit() on GPU."""

import numpy as np
import pytest
import tensorflow as tf

pytestmark = pytest.mark.skipif(
    not tf.config.list_physical_devices("GPU"), reason="GPU only"
)


def test_conv2d_model_fit():
    tf.keras.utils.set_random_seed(42)

    x = np.random.rand(256, 28, 28, 1).astype("float32")
    y = np.random.randint(0, 10, size=(256,)).astype("int64")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    history = model.fit(x, y, batch_size=128, epochs=1, verbose=0)

    assert history.history["loss"], "no loss recorded"
    assert np.isfinite(history.history["loss"][-1]), "non-finite training loss"
