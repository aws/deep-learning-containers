"""Shared fixtures and helpers for XGBoost SageMaker E2E tests.

Replaces ai_algorithms_qa orchestration with direct SageMaker SDK calls.
"""

import logging
import time

import boto3
import pytest
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from test_utils import random_suffix_name

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

E2E_TEST_BUCKET = "amazonai-algorithms-integration-tests"
E2E_DATA_PREFIX = "input/xgboost"

# Track created resources for cleanup
_created_models = []
_created_endpoints = []


def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"


def data_uri(key):
    return s3_uri(E2E_TEST_BUCKET, f"{E2E_DATA_PREFIX}/{key}")


def cleanup_resources():
    """Delete all SageMaker resources created during the test session."""
    sm = boto3.client("sagemaker")
    for ep in _created_endpoints:
        try:
            sm.delete_endpoint(EndpointName=ep)
        except Exception:
            pass
        try:
            sm.delete_endpoint_config(EndpointConfigName=ep)
        except Exception:
            pass
    for model_name in _created_models:
        try:
            sm.delete_model(ModelName=model_name)
        except Exception:
            pass
    _created_endpoints.clear()
    _created_models.clear()


@pytest.fixture(autouse=True, scope="session")
def _cleanup_after_session():
    """Automatically clean up all SageMaker resources after the test session."""
    yield
    cleanup_resources()


def run_training_job(
    image_uri,
    role,
    hyperparameters,
    train_s3_key,
    validation_s3_key,
    content_type,
    test_name="train",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size=10,
    max_run=1800,
    input_mode="File",
    train_distribution="ShardedByS3Key",
    checkpoint_s3_uri=None,
    enable_network_isolation=False,
    extra_channels=None,
):
    """Launch a SageMaker training job and return (job_name, duration, description)."""
    job_name = random_suffix_name(f"xgb-{test_name}", 32)
    output_path = s3_uri(E2E_TEST_BUCKET, f"e2e-output/{job_name}")

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

    if extra_channels:
        for name, uri in extra_channels.items():
            channels[name] = TrainingInput(s3_data=uri)

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
    LOGGER.info(
        f"Job {job_name} completed in {duration:.0f}s — status: {desc['TrainingJobStatus']}"
    )
    return job_name, duration, desc


def deploy_endpoint(
    image_uri, role, model_data, test_name="ep", instance_type="ml.m5.xlarge", env=None
):
    """Deploy a real-time endpoint and return (predictor, endpoint_name, model_name)."""
    from sagemaker.predictor import Predictor

    endpoint_name = random_suffix_name(f"xgb-{test_name}", 32)
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=role,
        env=env,
    )
    try:
        model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
    except Exception:
        # model may have been created even if deploy failed
        if model.name:
            _created_models.append(model.name)
        raise
    _created_models.append(model.name)
    _created_endpoints.append(endpoint_name)
    predictor = Predictor(endpoint_name=endpoint_name)
    return predictor, endpoint_name


def delete_endpoint(endpoint_name):
    """Delete endpoint, endpoint config, and associated model."""
    sm = boto3.client("sagemaker")
    # Find and delete the model associated with this endpoint
    try:
        ep_config = sm.describe_endpoint_config(EndpointConfigName=endpoint_name)
        for variant in ep_config.get("ProductionVariants", []):
            model_name = variant.get("ModelName")
            if model_name:
                try:
                    sm.delete_model(ModelName=model_name)
                except Exception:
                    pass
                if model_name in _created_models:
                    _created_models.remove(model_name)
    except Exception:
        pass
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
    except Exception:
        pass
    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except Exception:
        pass
    if endpoint_name in _created_endpoints:
        _created_endpoints.remove(endpoint_name)


def run_batch_transform(
    image_uri,
    role,
    model_data,
    input_s3_uri,
    content_type,
    test_name="bt",
    instance_type="ml.m5.xlarge",
    split_type="Line",
    accept="text/csv",
    env=None,
):
    """Run a batch transform job and return the job description."""
    job_name = random_suffix_name(f"xgb-{test_name}", 32)
    output_path = s3_uri(E2E_TEST_BUCKET, f"e2e-output/{job_name}")

    model = Model(image_uri=image_uri, model_data=model_data, role=role, env=env)
    model.create(instance_type=instance_type)
    _created_models.append(model.name)

    transformer = Transformer(
        model_name=model.name,
        instance_count=1,
        instance_type=instance_type,
        output_path=output_path,
        accept=accept,
    )
    try:
        transformer.transform(
            data=input_s3_uri,
            content_type=content_type,
            split_type=split_type,
            job_name=job_name,
        )
        transformer.wait()
    except Exception:
        sm = boto3.client("sagemaker")
        try:
            sm.stop_transform_job(TransformJobName=job_name)
        except Exception:
            pass
        raise

    sm = boto3.client("sagemaker")
    desc = sm.describe_transform_job(TransformJobName=job_name)
    return desc
