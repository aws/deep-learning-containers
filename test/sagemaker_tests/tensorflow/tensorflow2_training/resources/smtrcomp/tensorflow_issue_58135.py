# Inspired by https://github.com/tensorflow/tensorflow/issues/58135


import tensorflow as tf
import numpy as np
import argparse


class Chooser(tf.keras.layers.Layer):
    @tf.function
    def call(self, options_input, choices_input_logits):
        choices = tf.nn.softmax(choices_input_logits, axis=2)

        result = tf.linalg.matmul(choices, options_input)
        return result

def get_model():
    options_input = tf.keras.layers.Input(shape=(10,3), name="options")
    choices_input = tf.keras.layers.Input(shape=(5,10), name="choices")

    net = Chooser()(options_input, choices_input)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(1)(net)

    return tf.keras.Model([options_input, choices_input], net)


parser = argparse.ArgumentParser()
parser.add_argument('--jit_compile', type=bool, default=False)
args, unknown = parser.parse_known_args()


model = get_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.MeanAbsoluteError(),
    jit_compile=args.jit_compile
)

def batch_gen():
    while True:
        o = np.random.uniform(low=-1.0, high=1.0, size=(10, 3))
        c = np.random.uniform(low=0.0, high=1.0, size=(5, 10))

        y = 1

        yield {"options": o, "choices": c}, y

dataset = tf.data.Dataset.from_generator(batch_gen, output_types=({"options": tf.float32, "choices": tf.float32}, tf.float32))
dataset = dataset.batch(32)

model.fit(dataset, steps_per_epoch=100, epochs=1)