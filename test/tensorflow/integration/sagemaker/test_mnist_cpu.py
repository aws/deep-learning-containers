"""SageMaker CPU training integration tests for TensorFlow DLC.

Mirrors master's `test_mnist.py` coverage, translated to SDK v3
(ModelTrainer + InputData + SourceCode):

  - test_mnist_single_node_cpu          single-host, plain Keras
  - test_mnist_multi_host_no_strategy_cpu  2-host, plain Keras (mirrors
                                        master's `test_distributed_mnist_no_ps`
                                        — each host trains independently
                                        from the same input channel; no
                                        coordination, just smoke proves the
                                        2-host launcher path)
  - test_mnist_distributed_mwms_cpu     2-host, MultiWorkerMirroredStrategy
                                        (RING). Currently @pytest.mark.skip
                                        because SDK v3 + MWMS PerReplica
                                        distribution gap blocks it; see
                                        project memory + TODO follow-up.

We deliberately avoid SDK v3's MPI() distribution: its mpi_driver passes
process_count_per_node directly as `-np` without multiplying by host_count,
so multi-node never gets the intended global rank count.
"""

import os

import boto3
import pytest
from sagemaker import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
SOURCE_DIR = os.path.join(RESOURCE_DIR, "scripts")
MNIST_DATA_DIR = os.path.join(RESOURCE_DIR, "mnist", "data")
INSTANCE_TYPE = "ml.c5.xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]
DEFAULT_REGION = "us-west-2"


def _upload_mnist_data(key_prefix="scriptmode/mnist"):
    """Upload the bundled .npy MNIST subset to S3 and return the resulting URI.

    Mirrors master's `estimator.sagemaker_session.upload_data(...)` pattern.
    The Session is constructed with an explicit region because CI runners
    don't always have a default boto region configured.
    """
    sagemaker_session = Session(boto3.session.Session(region_name=DEFAULT_REGION))
    return sagemaker_session.upload_data(path=MNIST_DATA_DIR, key_prefix=key_prefix)


def _run_sm_training(
    image_uri,
    entry_script,
    source_dir,
    instance_type,
    instance_count,
    hyperparameters=None,
    input_data=None,
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
        # No MPI/torchrun launcher — SageMaker spawns one process per host.
        # For MWMS, the entry script builds TF_CONFIG itself.
        distributed=None,
    )

    model_trainer.train(input_data_config=input_data, wait=True)


def test_mnist_single_node_cpu():
    """Single-node CPU training with plain Keras.

    Mirrors master's `test_mnist`. The bundled .npy subset is uploaded to
    S3 and surfaced to the container at SM_CHANNEL_TRAINING."""
    inputs_s3 = _upload_mnist_data()
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        hyperparameters={"epochs": "1", "strategy": "none"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-cpu",
    )


def test_mnist_multi_host_no_strategy_cpu():
    """2-host CPU training with NO distribution strategy.

    Mirrors master's `test_distributed_mnist_no_ps` — runs the same plain
    Keras script on each host independently. There's no collective op or
    parameter coordination; the test exists to smoke-test the multi-host
    SageMaker launcher path with a TF DLC. Each host writes the same
    artifact, but only the chief saves to SM_MODEL_DIR (the script
    enforces the chief gate)."""
    inputs_s3 = _upload_mnist_data()
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"epochs": "1", "strategy": "none"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-2h-cpu",
    )


@pytest.mark.skip(
    reason=(
        "multi-node MultiWorkerMirroredStrategy + SDK v3 has a known PerReplica "
        "distribution gap. See project memory + TODO.md follow-up before "
        "re-enabling."
    )
)
def test_mnist_distributed_mwms_cpu():
    """2-node distributed CPU training with MultiWorkerMirroredStrategy (RING).

    Currently skipped — see decorator. The training script wires TF_CONFIG
    from SM_HOSTS so MWMS handles inter-host gRPC."""
    inputs_s3 = _upload_mnist_data()
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=INSTANCE_TYPE,
        instance_count=2,
        hyperparameters={"epochs": "1", "strategy": "mwms"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-mwms-cpu",
    )
