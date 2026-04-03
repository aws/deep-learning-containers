"""Pytest fixtures for XGBoost container tests.

Provides:
- --image flag for the container image URI
- Session-scoped S3 resource download
- Docker client fixture
"""

import logging
import os
import tempfile

import boto3
import pytest

import docker

LOGGER = logging.getLogger(__name__)

S3_BUCKET = "dlc-cicd-models"
S3_PREFIX = "xgboost/container_test_resources"


def pytest_addoption(parser):
    parser.addoption("--image", required=True, help="Docker image URI to test")


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image")


@pytest.fixture(scope="session")
def docker_client():
    return docker.from_env()


@pytest.fixture(scope="session")
def test_resources():
    """Download training/ and inference/ from S3 once per session."""
    tmpdir = tempfile.mkdtemp(prefix="xgb-container-test-")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = os.path.relpath(key, S3_PREFIX)
            if rel == ".":
                continue
            dest = os.path.join(tmpdir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            LOGGER.info("Downloading s3://%s/%s -> %s", S3_BUCKET, key, dest)
            s3.download_file(S3_BUCKET, key, dest)

    return tmpdir


@pytest.fixture(scope="session")
def training_resources(test_resources):
    return os.path.join(test_resources, "training")


@pytest.fixture(scope="session")
def inference_resources(test_resources):
    return os.path.join(test_resources, "inference")
