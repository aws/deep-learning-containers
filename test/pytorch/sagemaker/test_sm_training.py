"""SageMaker training integration tests for PyTorch DLC.

Uses SageMaker Python SDK v3 (ModelTrainer API).

Minimal test set covering P0 functionalities:
  F1: Distributed training with nccl (test_mnist_distributed_gpu)
  F2: torch.distributed primitives (test_dist_operations_gpu)

Tests launch real SageMaker training jobs — no GPU needed on the runner.
"""

import os

from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from sagemaker.train.distributed import Torchrun
from test_utils import random_suffix_name
from test_utils.constants import SAGEMAKER_ROLE

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
TRAINING_DATA_DIR = os.path.join(RESOURCE_DIR, "data", "training")
INSTANCE_TYPE = "ml.g4dn.12xlarge"


def _run_sm_training(
    image_uri,
    entry_script,
    source_dir,
    instance_type,
    instance_count,
    hyperparameters=None,
    job_name_prefix="pt-sm-test",
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
        role=SAGEMAKER_ROLE,
        base_job_name=random_suffix_name(job_name_prefix, 32),
        hyperparameters=hyperparameters or {},
        distributed=Torchrun() if instance_count > 1 else None,
    )

    input_data = None
    if os.path.isdir(TRAINING_DATA_DIR):
        input_data = [
            InputData(
                channel_name="training",
                data_source=TRAINING_DATA_DIR,
            )
        ]

    model_trainer.train(input_data_config=input_data, wait=True)


def test_mnist_distributed_gpu(image_uri, region):
    """F1: 2-node distributed GPU training with nccl backend."""
    _run_sm_training(
        image_uri=image_uri,
        entry_script="mnist.py",
        source_dir=RESOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"backend": "nccl", "epochs": "1"},
        job_name_prefix="pt-mnist-nccl",
    )


def test_dist_operations_gpu(image_uri, region):
    """F2: torch.distributed primitives (all_reduce, broadcast, etc.) on GPU."""
    _run_sm_training(
        image_uri=image_uri,
        entry_script="distributed_operations.py",
        source_dir=RESOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"backend": "nccl"},
        job_name_prefix="pt-dist-ops",
    )
