"""SageMaker CPU training integration tests for PyTorch DLC.

Uses SageMaker Python SDK v3 (ModelTrainer API) with gloo backend.
"""

import os

from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from sagemaker.train.distributed import Torchrun
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
INSTANCE_TYPE = "ml.c5.xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]
MNIST_S3_URI = "s3://dlc-cicd-models/pytorch/sagemaker-test-data/"


def _run_sm_training(
    image_uri,
    entry_script,
    source_dir,
    instance_type,
    instance_count,
    hyperparameters=None,
    input_data=None,
    job_name_prefix="pt-cpu-test",
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
        distributed=Torchrun() if instance_count > 1 else None,
    )

    model_trainer.train(input_data_config=input_data, wait=True)


def test_mnist_distributed_cpu():
    """2-node distributed CPU training with gloo backend."""
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=RESOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"backend": "gloo", "epochs": "1"},
        input_data=[
            InputData(
                channel_name="training",
                data_source=MNIST_S3_URI,
            )
        ],
        job_name_prefix="pt-mnist-gloo",
    )
