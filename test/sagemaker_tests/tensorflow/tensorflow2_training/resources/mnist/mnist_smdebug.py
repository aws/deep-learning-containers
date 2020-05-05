import argparse
import json
import os
import sys

import numpy as np
import tensorflow.compat.v2 as tf

import smdebug.tensorflow as smd
from smdebug.core.collection import CollectionKeys
from smdebug.core.reduction_config import ALLOWED_NORMS, ALLOWED_REDUCTIONS
from smdebug.tensorflow import ReductionConfig, SaveConfig
from smdebug.trials import create_trial


def _parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=1)
    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])

    return parser.parse_known_args()


def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, 'train', 'x_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'train', 'y_train.npy'))
    return x_train, y_train


def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, 'test', 'x_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'test', 'y_test.npy'))
    return x_test, y_test


def create_smdebug_hook(out_dir):
    include_collections = [
        CollectionKeys.WEIGHTS,
        CollectionKeys.BIASES,
        CollectionKeys.GRADIENTS,
        CollectionKeys.LOSSES,
        CollectionKeys.OUTPUTS,
        CollectionKeys.METRICS,
        CollectionKeys.LOSSES,
        CollectionKeys.OPTIMIZER_VARIABLES,
    ]
    save_config = SaveConfig(save_interval=3)
    hook = smd.KerasHook(
        out_dir,
        save_config=save_config,
        include_collections=include_collections,
        reduction_config=ReductionConfig(norms=ALLOWED_NORMS, reductions=ALLOWED_REDUCTIONS),
    )
    return hook


args, unknown = _parse_args()

hook = create_smdebug_hook(args.smdebug_path)
hooks = [hook]

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
x_train, y_train = _load_training_data(args.train)
x_test, y_test = _load_testing_data(args.train)
model.fit(x_train, y_train, epochs=args.epochs, callbacks=hooks)
model.evaluate(x_test, y_test, callbacks=hooks)

if args.current_host == args.hosts[0]:
    model.save(os.path.join('/opt/ml/model', 'my_model.h5'))

print("Created the trial with out_dir {0}".format(args.smdebug_path))
trial = create_trial(args.smdebug_path)
assert trial

print(f"trial.tensor_names() = {trial.tensor_names()}")

weights_tensors = hook.collection_manager.get("weights").tensor_names
assert len(weights_tensors) > 0

losses_tensors = hook.collection_manager.get("losses").tensor_names
assert len(losses_tensors) > 0
