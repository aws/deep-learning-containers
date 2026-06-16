"""SageMaker GPU training integration tests for TensorFlow DLC.

Uses SDK v3 (ModelTrainer + InputData + SourceCode):

  - test_mnist_single_node_gpu          single-host single-GPU, plain Keras
  - test_mnist_multi_host_no_strategy_gpu  2-host, plain Keras (each host
                                        trains independently; smoke-tests
                                        the multi-host launcher path on
                                        the GPU image)
  - test_mnist_mirrored_strategy_gpu    single-host multi-GPU with
                                        tf.distribute.MirroredStrategy
                                        (one box, all GPUs)
  - test_mnist_distributed_mwms_gpu     2-host, MultiWorkerMirroredStrategy
                                        (NCCL). On TF 2.21 / Keras 3,
                                        model.fit() under MWMS hits a
                                        PerReplica distribution gap, so the
                                        entry script uses a custom training
                                        loop (strategy.run + reduce).

We deliberately avoid SDK v3's MPI() distribution: its mpi_driver passes
process_count_per_node directly as `-np` without multiplying by host_count,
so multi-node never gets the intended global rank count.

Tests launch real SageMaker training jobs — no GPU needed on the runner.
"""

import os
from urllib.parse import urlparse

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Compute, InputData, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

RESOURCE_DIR = os.path.join(os.path.dirname(__file__), "resources")
SOURCE_DIR = os.path.join(RESOURCE_DIR, "scripts")
MNIST_DATA_DIR = os.path.join(RESOURCE_DIR, "mnist", "data")
SINGLE_GPU_INSTANCE = "ml.g4dn.xlarge"
# DLC test account has ml.g4dn.xlarge quota=1; multi-host needs 2 instances.
# ml.g4dn.2xlarge has quota=8, same T4 GPU.
MULTI_HOST_GPU_INSTANCE = "ml.g4dn.2xlarge"
MULTI_GPU_INSTANCE = "ml.g4dn.12xlarge"
IMAGE_URI = os.environ["TEST_IMAGE_URI"]
DEFAULT_REGION = "us-west-2"


def _upload_mnist_data(key_prefix="scriptmode/mnist"):
    """Upload the bundled .npy MNIST subset to S3 and return the URI."""
    sagemaker_session = Session(boto3.session.Session(region_name=DEFAULT_REGION))
    return sagemaker_session.upload_data(path=MNIST_DATA_DIR, key_prefix=key_prefix)


def _run_sm_training(
    image_uri,
    entry_script,
    source_dir,
    instance_type,
    instance_count,
    hyperparameters=None,
    environment=None,
    input_data=None,
    job_name_prefix="tf-gpu-test",
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
        # No MPI/torchrun launcher — SageMaker spawns one process per host.
        # MirroredStrategy uses all visible GPUs in that one process; MWMS
        # would also coordinate across hosts via TF_CONFIG (see skipped test).
        distributed=None,
    )

    model_trainer.train(input_data_config=input_data, wait=True)
    return model_trainer


def _assert_s3_file_exists(region, s3_url):
    """Verify that the given s3:// URL points to an existing object.

    head-object via boto3 raises if the key is missing, which surfaces as a
    clear test failure when SageMaker didn't upload the model artifact."""
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource("s3", region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip("/")).load()


def test_mnist_single_node_gpu():
    """Single-host single-GPU training with plain Keras (no strategy).

    TF picks up the one visible GPU automatically without any explicit
    distribute scope."""
    inputs_s3 = _upload_mnist_data()
    model_trainer = _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=SINGLE_GPU_INSTANCE,
        instance_count=1,
        hyperparameters={"strategy": "none"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-gpu",
    )
    _assert_s3_file_exists(
        DEFAULT_REGION, model_trainer._latest_training_job.model_artifacts.s3_model_artifacts
    )


def test_mnist_multi_host_no_strategy_gpu():
    """2-host single-GPU training with NO distribution strategy.

    Each host trains independently — no NCCL, no TF_CONFIG. Smoke-tests the
    GPU multi-host launcher path."""
    inputs_s3 = _upload_mnist_data()
    model_trainer = _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=MULTI_HOST_GPU_INSTANCE,
        instance_count=2,
        hyperparameters={"strategy": "none"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-2h-gpu",
    )
    _assert_s3_file_exists(
        DEFAULT_REGION, model_trainer._latest_training_job.model_artifacts.s3_model_artifacts
    )


def test_mnist_mirrored_strategy_gpu():
    """Single-host multi-GPU training with tf.distribute.MirroredStrategy.

    Covers the common DLC scenario of one box with all GPUs. MirroredStrategy
    doesn't need TF_CONFIG and uses NCCL all-reduce across the local GPUs
    only — keeps the test cheap (one instance) while exercising the GPU
    collective path."""
    inputs_s3 = _upload_mnist_data()
    model_trainer = _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=MULTI_GPU_INSTANCE,
        instance_count=1,
        hyperparameters={"strategy": "mirrored"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-mirrored-gpu",
    )
    _assert_s3_file_exists(
        DEFAULT_REGION, model_trainer._latest_training_job.model_artifacts.s3_model_artifacts
    )


def test_mnist_distributed_mwms_gpu():
    """2-node distributed GPU training with MultiWorkerMirroredStrategy (NCCL).

    The training script uses a custom strategy.run loop on TF 2.21 / Keras 3
    (model.fit() hits a PerReplica distribution gap under MWMS). If NCCL
    fails to initialise or all_reduce hangs, the SageMaker job fails and
    pytest surfaces that here."""
    inputs_s3 = _upload_mnist_data()
    _run_sm_training(
        image_uri=IMAGE_URI,
        entry_script="mnist.py",
        source_dir=SOURCE_DIR,
        instance_type=MULTI_GPU_INSTANCE,
        instance_count=2,
        hyperparameters={"epochs": "2", "strategy": "mwms"},
        environment={"FI_EFA_FORK_SAFE": "1"},
        input_data=[InputData(channel_name="training", data_source=inputs_s3)],
        job_name_prefix="tf-mnist-mwms-gpu",
    )
