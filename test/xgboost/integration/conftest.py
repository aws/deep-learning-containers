"""Shared fixtures and helpers for XGBoost SageMaker integration tests.

Replaces ai_algorithms_qa orchestration with direct SageMaker SDK calls.
"""

import logging
import time

import boto3
import pytest
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, LibSMVSerializer
from sagemaker.transformer import Transformer
from test_utils import random_suffix_name

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

INTEG_TEST_BUCKET = "amazonai-algorithms-integ-tests"
INTEG_DATA_PREFIX = "input/xgboost"


def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"


def data_uri(key):
    return s3_uri(INTEG_TEST_BUCKET, f"{INTEG_DATA_PREFIX}/{key}")


def run_training_job(
    image_uri,
    role,
    hyperparameters,
    train_s3_key,
    validation_s3_key,
    content_type,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size=10,
    max_run=1800,
    input_mode="File",
    train_distribution="ShardedByS3Key",
    checkpoint_s3_uri=None,
    enable_network_isolation=False,
):
    """Launch a SageMaker training job and return (job_name, duration, description)."""
    job_name = random_suffix_name("xgb-integ", 50)
    output_path = s3_uri(INTEG_TEST_BUCKET, f"integ-output/{job_name}")

    estimator = Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=output_path,
        hyperparameters=hyperparameters,
        volume_size=volume_size,
        max_run=max_run,
        input_mode=input_mode,
        checkpoint_s3_uri=checkpoint_s3_uri,
        enable_network_isolation=enable_network_isolation,
    )

    channels = {
        "train": TrainingInput(
            s3_data=data_uri(train_s3_key),
            content_type=content_type,
            distribution=train_distribution,
        ),
        "validation": TrainingInput(
            s3_data=data_uri(validation_s3_key),
            content_type=content_type,
            distribution="FullyReplicated",
        ),
    }

    LOGGER.info(f"Starting job: {job_name} ({instance_count}x {instance_type})")
    sm = boto3.client("sagemaker")
    start = time.time()
    try:
        estimator.fit(channels, job_name=job_name)
    except Exception:
        try:
            sm.stop_training_job(TrainingJobName=job_name)
        except Exception:
            pass
        raise
    duration = time.time() - start

    desc = sm.describe_training_job(TrainingJobName=job_name)
    LOGGER.info(f"Job {job_name} completed in {duration:.0f}s — status: {desc['TrainingJobStatus']}")
    return job_name, duration, desc


def deploy_endpoint(image_uri, role, model_data, instance_type="ml.m5.xlarge", env=None):
    """Deploy a real-time endpoint and return (predictor, endpoint_name)."""
    endpoint_name = random_suffix_name("xgb-ep", 50)
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=role,
        env=env,
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    return predictor, endpoint_name


def delete_endpoint(endpoint_name):
    sm = boto3.client("sagemaker")
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
    except Exception:
        pass
    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except Exception:
        pass


def run_batch_transform(
    image_uri, role, model_data, input_s3_uri, content_type,
    instance_type="ml.m5.xlarge", split_type="Line", accept="text/csv",
):
    """Run a batch transform job and return the job description."""
    job_name = random_suffix_name("xgb-bt", 50)
    output_path = s3_uri(INTEG_TEST_BUCKET, f"integ-output/{job_name}")

    model = Model(image_uri=image_uri, model_data=model_data, role=role)
    model_name = random_suffix_name("xgb-model", 50)
    model.create(instance_type=instance_type)

    transformer = Transformer(
        model_name=model.name,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        accept=accept,
    )
    transformer.transform(
        data=input_s3_uri,
        content_type=content_type,
        split_type=split_type,
        job_name=job_name,
    )
    transformer.wait()

    sm = boto3.client("sagemaker")
    desc = sm.describe_transform_job(TransformJobName=job_name)
    return desc
