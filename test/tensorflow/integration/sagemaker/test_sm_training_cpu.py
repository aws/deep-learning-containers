"""SageMaker CPU training integration tests for TensorFlow DLC.

Uses SageMaker Python SDK v3 (ModelTrainer API). For multi-host distributed
training we rely on TF's native `MultiWorkerMirroredStrategy` which uses
TF_CONFIG + gRPC (no MPI). The training script reads SM_HOSTS / SM_CURRENT_HOST
and constructs TF_CONFIG itself; SageMaker spawns one process per host
automatically when distributed=None.

We deliberately avoid SDK v3's MPI() distribution: its mpi_driver passes
process_count_per_node directly as `-np` without multiplying by host_count,
so multi-node training never gets the intended global rank count.
"""

import os

from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
INSTANCE_TYPE = "ml.c5.xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]


def _run_sm_training(
    image_uri,
    entry_script,
    source_dir,
    instance_type,
    instance_count,
    hyperparameters=None,
    job_name_prefix="tf-cpu-test",
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
        # No MPI/torchrun launcher — SageMaker spawns one process per host;
        # the training script builds TF_CONFIG and lets MultiWorkerMirrored
        # Strategy handle inter-host gRPC.
        distributed=None,
    )

    model_trainer.train(wait=True)


def test_mnist_distributed_cpu():
    """2-node distributed CPU training with MultiWorkerMirroredStrategy (RING)."""
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=RESOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"epochs": "1"},
        job_name_prefix="tf-mnist-mwms-cpu",
    )
