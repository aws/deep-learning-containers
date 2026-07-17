"""Smoke test: train a Conv2D model via model.fit() on GPU."""

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

pytestmark = pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="GPU only")


def test_conv2d_model_fit_gpu():
    tf.keras.utils.set_random_seed(42)

    num_classes = 10
    input_shape = (28, 28, 1)

    x_train = tf.random.normal([256, *input_shape])
    y_train = tf.random.uniform([256], maxval=num_classes, dtype=tf.int64)
    y_train = keras.utils.to_categorical(y_train, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=0)

    final_loss = history.history["loss"][-1]
    assert np.isfinite(final_loss), f"training produced non-finite loss: {final_loss}"
