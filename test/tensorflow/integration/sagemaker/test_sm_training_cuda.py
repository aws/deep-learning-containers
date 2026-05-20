"""SageMaker training integration tests for TensorFlow DLC.

Uses SageMaker Python SDK v3 (ModelTrainer API).

Minimal test set covering P0 functionalities:
  F1: Distributed training (test_mnist_distributed_gpu) — uses
      `MultiWorkerMirroredStrategy` driven by TF_CONFIG (no MPI launcher).
      MultiWorkerMirroredStrategy auto-selects NCCL on multi-GPU hosts.

We deliberately avoid SDK v3's MPI() distribution: its mpi_driver passes
process_count_per_node directly as `-np` without multiplying by host_count,
so multi-node never gets the intended global rank count. SageMaker spawns
one process per host when distributed=None; that one process then sees all
visible GPUs as a single MultiWorkerMirrored worker.

Tests launch real SageMaker training jobs — no GPU needed on the runner.
"""

import os

from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
INSTANCE_TYPE = "ml.g4dn.12xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]


def _run_sm_training(
    image_uri,
    entry_script,
    source_dir,
    instance_type,
    instance_count,
    hyperparameters=None,
    environment=None,
    job_name_prefix="tf-sm-test",
):
    """Launch a SageMaker training job using ModelTrainer (SDK v3) and wait for completion."""
    source_code = SourceCode(
        source_dir=source_dir,
        entry_script=entry_script,
    )

    compute = Compute(
        instance_type=instance_type,
        instance_count=instance_count,
    )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        source_code=source_code,
        compute=compute,
        role=os.environ.get("SM_ROLE_ARN"),
        base_job_name=random_suffix_name(job_name_prefix, 32),
        hyperparameters=hyperparameters or {},
        environment=environment or {},
        # No MPI/torchrun launcher — SageMaker spawns one process per host;
        # the training script builds TF_CONFIG and lets MultiWorkerMirrored
        # Strategy use all visible GPUs on each host as a single worker.
        distributed=None,
    )

    model_trainer.train(wait=True)


def test_mnist_distributed_gpu():
    """F1: 2-node distributed GPU training with MultiWorkerMirroredStrategy (NCCL).

    The training script asserts final accuracy > 0.5 on its own — if NCCL
    fails to initialise or all_reduce calls hang, the SageMaker job fails
    and pytest surfaces that here. We don't post-process logs in this PR
    (the deferred multi-node-EFA test in TODO.md does NCCL transport
    verification with `NET/OFI` log parsing)."""
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=RESOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"epochs": "1"},
        environment={"FI_EFA_FORK_SAFE": "1"},
        job_name_prefix="tf-mnist-mwms-gpu",
    )
