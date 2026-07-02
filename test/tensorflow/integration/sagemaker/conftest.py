"""Shared fixtures for TF DLC SageMaker integration tests.

Uses SDK v3 (ModelTrainer + InputData + SourceCode). We deliberately avoid
SDK v3's MPI() distribution: its mpi_driver passes process_count_per_node
directly as `-np` without multiplying by host_count, so multi-node never
gets the intended global rank count.
"""

import os
from urllib.parse import urlparse

import boto3
import pytest
from sagemaker.core.training.configs import Compute, SourceCode
from sagemaker.train import ModelTrainer
from test_utils import random_suffix_name

# ── Path constants ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def resource_dir():
    return os.path.join(os.path.dirname(__file__), "resources")


@pytest.fixture(scope="session")
def source_dir(resource_dir):
    return os.path.join(resource_dir, "scripts")


# ── Environment / config fixtures ──────────────────────────────────────────


@pytest.fixture(scope="session")
def image_uri():
    return os.environ["TEST_IMAGE_URI"]


@pytest.fixture(scope="session")
def default_region():
    return "us-west-2"


@pytest.fixture(scope="session")
def mnist_s3_uri():
    return "s3://dlc-cicd-models/tensorflow/sagemaker-test-data/MNIST/"


# ── Helper factories (return callables) ────────────────────────────────────


@pytest.fixture(scope="session")
def sm_trainer(image_uri, source_dir):
    """Return a callable that launches a SageMaker training job via ModelTrainer (SDK v3)."""

    def _run(
        entry_script,
        instance_type,
        instance_count,
        hyperparameters=None,
        environment=None,
        input_data=None,
        job_name_prefix="tf-sm-test",
    ):
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
            # For MWMS, the entry script builds TF_CONFIG itself.
            distributed=None,
        )

        model_trainer.train(input_data_config=input_data, wait=True)
        return model_trainer

    return _run


@pytest.fixture(scope="session")
def assert_s3_file_exists(default_region):
    """Return a callable that asserts the given s3:// URL points to an existing object.

    head-object via boto3 raises if the key is missing, which surfaces as a
    clear test failure when SageMaker didn't upload the model artifact."""

    def _assert(s3_url):
        parsed_url = urlparse(s3_url)
        s3 = boto3.resource("s3", region_name=default_region)
        s3.Object(parsed_url.netloc, parsed_url.path.lstrip("/")).load()

    return _assert
