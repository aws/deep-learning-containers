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

"""MNIST distributed training entry point for SageMaker integration tests.

Runs `MultiWorkerMirroredStrategy` over MPI launched by SageMaker's `Mpi()`
distribution. Each MPI rank corresponds to one SageMaker host; MPI launches a
single TF process per host (one rank per node).

TF_CONFIG plumbing
------------------
TF needs a TF_CONFIG env var of the form
  {"cluster": {"worker": ["host1:port", "host2:port"]}, "task": {"type": "worker", "index": <rank>}}
on every worker before strategy initialisation.

We build it from the SageMaker-provided env vars:
  SM_HOSTS         — JSON list of host names, ordered (e.g. '["algo-1","algo-2"]')
  SM_CURRENT_HOST  — this host's name
  OMPI_COMM_WORLD_RANK — set by mpirun, gives this process's rank (0-indexed)

We prefer SM_HOSTS for cluster-shape (deterministic ordering across nodes), and
OMPI_COMM_WORLD_RANK for the local task index when present (it's the source of
truth that mpirun used for placement). When MPI is not active (single-node
fallback), we fall back to SM_HOSTS.index(SM_CURRENT_HOST).
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


def _build_tf_config():
    """Construct the TF_CONFIG dict from SageMaker + MPI env vars."""
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]

    # Prefer the rank mpirun assigned; fall back to the position in SM_HOSTS.
    rank_env = os.environ.get("OMPI_COMM_WORLD_RANK")
    if rank_env is not None:
        task_index = int(rank_env)
    else:
        task_index = hosts.index(current_host)

    cluster = {"worker": [f"{host}:{WORKER_PORT}" for host in hosts]}
    return {"cluster": cluster, "task": {"type": "worker", "index": task_index}}


def _load_mnist_subset(num_samples=1000):
    """Download MNIST via tf.keras (each worker pulls independently). Use a
    small subset so the test stays fast — the bar is `accuracy > 0.5`."""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:num_samples].astype("float32") / 255.0
    x_train = x_train[..., None]  # (N, 28, 28) -> (N, 28, 28, 1)
    y_train = y_train[:num_samples].astype("int64")
    return x_train, y_train


def _build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def _log_diagnostics():
    """Log all SM/MPI/TF env vars + library versions so a CloudWatch reader
    can diagnose distributed training failures from the log alone. This runs
    BEFORE strategy init so we capture state even when MultiWorkerMirroredStrategy
    crashes during construction."""
    logger.info("=== mnist.py diagnostics ===")
    logger.info("Python: %s", sys.version)
    logger.info("TensorFlow: %s", tf.__version__)
    logger.info("TF built with CUDA: %s", tf.test.is_built_with_cuda())
    logger.info("Physical GPUs: %s", tf.config.list_physical_devices("GPU"))
    logger.info("Physical CPUs: %s", tf.config.list_physical_devices("CPU"))

    sm_keys = sorted(k for k in os.environ if k.startswith("SM_"))
    mpi_keys = sorted(k for k in os.environ if k.startswith(("OMPI_", "PMIX_", "PMI_")))
    logger.info("SM_* env vars: %s", {k: os.environ[k] for k in sm_keys})
    logger.info("MPI env vars: %s", {k: os.environ[k] for k in mpi_keys})

    try:
        import mpi4py
        from mpi4py import MPI as _mpi  # noqa: N811

        logger.info("mpi4py: %s", mpi4py.__version__)
        logger.info(
            "MPI rank %d of %d on %s",
            _mpi.COMM_WORLD.Get_rank(),
            _mpi.COMM_WORLD.Get_size(),
            _mpi.Get_processor_name(),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("mpi4py import/MPI init failed: %r", e)
    logger.info("=== end diagnostics ===")


def train(args):
    _log_diagnostics()

    is_distributed = len(json.loads(os.environ.get("SM_HOSTS", "[]"))) > 1

    if is_distributed:
        tf_config = _build_tf_config()
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        logger.info("TF_CONFIG=%s", os.environ["TF_CONFIG"])

    # Default communication is auto: NCCL on GPU, RING on CPU.
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    logger.info("Strategy initialised. num_replicas_in_sync=%d", strategy.num_replicas_in_sync)

    x_train, y_train = _load_mnist_subset(args.num_samples)

    global_batch_size = args.batch_size * max(strategy.num_replicas_in_sync, 1)
    dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .batch(global_batch_size)
        .repeat()
    )
    steps_per_epoch = max(len(x_train) // global_batch_size, 1)

    with strategy.scope():
        model = _build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    history = model.fit(dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch, verbose=2)
    final_acc = float(history.history["accuracy"][-1])
    logger.info("Final training accuracy: %.4f", final_acc)
    assert final_acc > 0.5, f"training accuracy {final_acc:.4f} below 0.5 threshold"

    # Only the chief saves the model (avoids cross-worker write races).
    if not is_distributed or os.environ.get("OMPI_COMM_WORLD_RANK", "0") == "0":
        model_dir = os.environ.get("SM_MODEL_DIR", args.model_dir)
        model.save(os.path.join(model_dir, "mnist_model.keras"))
        logger.info("Model saved to %s", model_dir)


if __name__ == "__main__":
    print("sys.argv: ", sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )

    args = parser.parse_args()
    train(args)
