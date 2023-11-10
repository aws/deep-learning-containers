import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf

import smppy


def _parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=1)
    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--save-as-tf", type=bool, default=False)

    return parser.parse_known_args()


def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, "train", "x_train.npy"))
    y_train = np.load(os.path.join(base_dir, "train", "y_train.npy"))
    return x_train, y_train


def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, "test", "x_test.npy"))
    y_test = np.load(os.path.join(base_dir, "test", "y_test.npy"))
    return x_test, y_test


args, unknown = _parse_args()

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

os.makedirs("/opt/ml/output/profiler/framework", exist_ok=True)
smp = smppy.SMProfiler.instance()
config = smppy.Config()
config.profiler = {
    "EnableCuda": "1",
}
smp.configure(config)

smp.start_profiling()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
x_train, y_train = _load_training_data(args.train)
x_test, y_test = _load_testing_data(args.train)
with smppy.annotate("Training"):
    model.fit(x_train, y_train, epochs=args.epochs)
with smppy.annotate("Evaluation"):
    model.evaluate(x_test, y_test)

smp.stop_profiling()

smp_files = glob.glob("/opt/ml/output/profiler/framework/*.smpraw")
assert len(smp_files) > 0, "The local output folder doesn't contain any sagemaker profiler files"
for f in smp_files:
    assert os.path.getsize(f) > 0, "sagemaker profiler file has size 0"
