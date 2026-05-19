"""SageMaker CPU training integration tests for TensorFlow DLC.

Uses SageMaker Python SDK v3 (ModelTrainer API) with `MPI()` distribution +
`MultiWorkerMirroredStrategy` (RING collective on CPU).
"""

import os

from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train import ModelTrainer
from sagemaker.train.distributed import MPI
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
        # CPU instances have 0 GPUs; MPI defaults process_count_per_node to GPU
        # count, so set to 1 explicitly to launch one rank per node.
        distributed=MPI(process_count_per_node=1) if instance_count > 1 else None,
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
