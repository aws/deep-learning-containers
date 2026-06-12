# Copyright 2018-2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""MNIST training entry script for SageMaker integration tests.

Loads MNIST `.npy` files from the `training` channel (SM_CHANNEL_TRAINING)
mirroring the master branch script. The same script is reused across all
test variants:

  STRATEGY=none      Plain Keras (single-host single-device, or per-host
                     independent training when instance_count > 1 — this
                     mirrors master's `test_distributed_mnist_no_ps` "no
                     parameter server" smoke test, which just runs the
                     identical plain script on each host).
  STRATEGY=mirrored  tf.distribute.MirroredStrategy — single-host
                     multi-GPU training. Selects all visible GPUs.
  STRATEGY=mwms      tf.distribute.MultiWorkerMirroredStrategy — multi-
                     host distributed training. Builds TF_CONFIG from
                     SM_HOSTS / SM_CURRENT_HOST so SageMaker can spawn
                     one process per host (no MPI launcher).

This is a deployability smoke test, not an accuracy gate: the script
trains, exports a SavedModel, and exits. Artifact existence is verified
by the test layer (mirroring master's `_assert_s3_file_exists`).
"""

from __future__ import absolute_import, print_function

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import tensorflow as tf

# Reproducibility — mirrors PT main's torch.manual_seed(1) pattern. Cuts down
# accuracy-bar flakes when training a tiny subset for one epoch.
os.environ.setdefault("PYTHONHASHSEED", "1")
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

WORKER_PORT = 12345


def _load_training_data(base_dir):
    x_train = np.load(os.path.join(base_dir, "train", "x_train.npy"))
    y_train = np.load(os.path.join(base_dir, "train", "y_train.npy"))
    return x_train, y_train


def _load_testing_data(base_dir):
    x_test = np.load(os.path.join(base_dir, "test", "x_test.npy"))
    y_test = np.load(os.path.join(base_dir, "test", "y_test.npy"))
    return x_test, y_test


def _build_tf_config():
    """One TF worker per SageMaker host. Each host has its own process and
    talks to peers over gRPC on WORKER_PORT."""
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    cluster = {"worker": [f"{h}:{WORKER_PORT}" for h in hosts]}
    return {"cluster": cluster, "task": {"type": "worker", "index": hosts.index(current_host)}}


def _build_model():
    """Same dense MLP as master's mnist.py — keeps the test fast and the
    accuracy bar achievable in 1 epoch on the 1000-sample subset."""
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )


def _log_diagnostics():
    """Log SM env vars + TF/GPU info so a CloudWatch reader can diagnose
    training failures from the log alone. Runs BEFORE strategy init so we
    capture state even when distribute.Strategy crashes during construction."""
    logger.info("=== mnist.py diagnostics ===")
    logger.info("Python: %s", sys.version)
    logger.info("TensorFlow: %s", tf.__version__)
    logger.info("TF built with CUDA: %s", tf.test.is_built_with_cuda())
    logger.info("Physical GPUs: %s", tf.config.list_physical_devices("GPU"))
    logger.info("Physical CPUs: %s", tf.config.list_physical_devices("CPU"))

    sm_keys = sorted(k for k in os.environ if k.startswith("SM_"))
    logger.info("SM_* env vars: %s", {k: os.environ[k] for k in sm_keys})
    logger.info("=== end diagnostics ===")


def _make_strategy(strategy_name):
    if strategy_name == "mirrored":
        # Single-host multi-GPU; selects all visible GPUs by default.
        return tf.distribute.MirroredStrategy()
    if strategy_name == "mwms":
        hosts = json.loads(os.environ.get("SM_HOSTS", "[]"))
        if len(hosts) > 1:
            os.environ["TF_CONFIG"] = json.dumps(_build_tf_config())
            logger.info("TF_CONFIG=%s", os.environ["TF_CONFIG"])
        # Communication is auto: NCCL on GPU, RING on CPU.
        return tf.distribute.MultiWorkerMirroredStrategy()
    return None


def _train_mwms(strategy, x_train, y_train, args):
    """Custom training loop for MultiWorkerMirroredStrategy.

    On TF 2.21 / Keras 3, `model.fit()` under MWMS hits a PerReplica
    distribution gap (Keras 3's compile/fit pipeline doesn't iterate the
    PerReplica values produced by the auto-distributed dataset). The
    workaround validated locally is a manual `strategy.run` loop with
    `distribute_datasets_from_function` — that path bypasses Keras's
    fit pipeline and drives gradients directly on each replica.

    Returns the final epoch's average per-replica loss and accuracy.
    """
    global_batch_size = args.batch_size * max(strategy.num_replicas_in_sync, 1)

    def dataset_fn(ctx):
        per_replica_batch = ctx.get_per_replica_batch_size(global_batch_size)
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # Shard across input pipelines so each worker sees disjoint data.
        ds = ds.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        return (
            ds.shuffle(len(x_train), seed=1)
            .repeat()
            .batch(per_replica_batch)
            .prefetch(tf.data.AUTOTUNE)
        )

    dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

    with strategy.scope():
        model = _build_model()
        optimizer = tf.keras.optimizers.Adam()
        # Eagerly build optimizer slot variables (Adam momentum/velocity) before
        # the @tf.function trace. Without this, slots are created lazily inside
        # the traced graph's cond branch, which fails cross-worker on multi-host
        # GPU with "must feed value for placeholder cond/Placeholder_*" — the
        # cross-worker variable-creation path can't fill in the deferred init.
        # CPU multi-host doesn't trip this because slot creation stays local.
        optimizer.build(model.trainable_variables)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

    @tf.function
    def train_step(inputs):
        x_batch, y_batch = inputs
        with tf.GradientTape() as tape:
            preds = model(x_batch, training=True)
            per_example_loss = loss_fn(y_batch, preds)
            # Scale by GLOBAL batch so gradients average correctly across replicas.
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_accuracy.update_state(y_batch, preds)
        return loss

    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    steps_per_epoch = max(len(x_train) // global_batch_size, 1)
    logger.info(
        "MWMS custom loop: global_batch=%d steps_per_epoch=%d epochs=%d",
        global_batch_size,
        steps_per_epoch,
        args.epochs,
    )

    it = iter(dist_dataset)
    final_loss = 0.0
    final_acc = 0.0
    for epoch in range(args.epochs):
        train_accuracy.reset_state()
        epoch_loss_sum = 0.0
        for step in range(steps_per_epoch):
            batch = next(it)
            step_loss = float(distributed_train_step(batch))
            epoch_loss_sum += step_loss
        final_loss = epoch_loss_sum / steps_per_epoch
        final_acc = float(train_accuracy.result())
        # Match the loss/accuracy log format the existing tuner regex expects.
        logger.info(
            "Epoch %d/%d - loss: %.4f - accuracy: %.4f",
            epoch + 1,
            args.epochs,
            final_loss,
            final_acc,
        )

    return model, final_loss, final_acc


def train(args):
    _log_diagnostics()

    hosts = json.loads(os.environ.get("SM_HOSTS", "[]"))
    current_host = os.environ.get("SM_CURRENT_HOST", hosts[0] if hosts else "")
    is_chief = (not hosts) or current_host == hosts[0]

    x_train, y_train = _load_training_data(args.train)
    x_test, y_test = _load_testing_data(args.train)

    strategy = _make_strategy(args.strategy)
    num_replicas = strategy.num_replicas_in_sync if strategy is not None else 1
    logger.info("Strategy=%s num_replicas_in_sync=%d", args.strategy, num_replicas)

    if args.strategy == "mwms":
        # MWMS uses a custom training loop — see _train_mwms for the why.
        model, _, final_acc = _train_mwms(strategy, x_train, y_train, args)
    elif strategy is None:
        model = _build_model()
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history = model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
        final_acc = float(history.history["accuracy"][-1])
    else:
        # MirroredStrategy: model.fit() works fine on a single host.
        global_batch_size = args.batch_size * max(num_replicas, 1)
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(len(x_train), seed=1)
            .repeat()
            .batch(global_batch_size)
        )

        with strategy.scope():
            model = _build_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

        steps_per_epoch = max(len(x_train) // global_batch_size, 1)
        history = model.fit(
            train_ds, epochs=args.epochs, steps_per_epoch=steps_per_epoch, verbose=2
        )
        final_acc = float(history.history["accuracy"][-1])

    logger.info("Final training accuracy: %.4f", final_acc)

    # Eval is best-effort — single-host plain Keras only, since MWMS scope
    # complicates evaluate() with the tiny dataset.
    if strategy is None:
        loss, acc = model.evaluate(x_test, y_test, verbose=2)
        logger.info("Eval loss=%.4f acc=%.4f", loss, acc)

    # Only the chief saves — avoids cross-worker write races to the same
    # /opt/ml/model/ path. Save under "1" so the artifact follows the
    # SavedModel layout master uses (model_dir/<version>).
    if is_chief:
        model_dir = args.model_dir
        save_path = os.path.join(model_dir, "1")
        model.export(save_path)
        logger.info("Model saved to %s", save_path)


if __name__ == "__main__":
    print("sys.argv: ", sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--strategy",
        type=str,
        default=os.environ.get("STRATEGY", "none"),
        choices=["none", "mirrored", "mwms"],
        help="tf.distribute strategy: none | mirrored | mwms",
    )
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )

    args, _ = parser.parse_known_args()
    train(args)
