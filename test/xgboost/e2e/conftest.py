"""Shared fixtures and helpers for XGBoost SageMaker E2E tests.

Replaces ai_algorithms_qa orchestration with direct SageMaker SDK calls.
"""

import logging
import time

import boto3
import pytest
from sagemaker.serve.model_builder import InferenceSpec, ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    CheckpointConfig,
    Compute,
    InputData,
    Networking,
    OutputDataConfig,
    StoppingCondition,
)
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

    compute = Compute(
        instance_type=instance_type,
        instance_count=instance_count,
        volume_size_in_gb=volume_size,
    )

    trainer_kwargs = dict(
        training_image=image_uri,
        role=role,
        compute=compute,
        hyperparameters=hyperparameters,
        output_data_config=OutputDataConfig(s3_output_path=output_path),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=max_run),
        training_input_mode=input_mode,
    )

    if checkpoint_s3_uri:
        trainer_kwargs["checkpoint_config"] = CheckpointConfig(
            s3_uri=checkpoint_s3_uri,
        )

    if enable_network_isolation:
        trainer_kwargs["networking"] = Networking(
            enable_network_isolation=True,
        )

    trainer = ModelTrainer(**trainer_kwargs)

    input_data_config = [
        InputData(
            channel_name="train",
            data_source=data_uri(train_s3_key),
            content_type=content_type,
        ),
        InputData(
            channel_name="validation",
            data_source=data_uri(validation_s3_key),
            content_type=content_type,
        ),
    ]

    if extra_channels:
        for name, uri in extra_channels.items():
            input_data_config.append(InputData(channel_name=name, data_source=uri))

    LOGGER.info(f"Starting job: {job_name} ({instance_count}x {instance_type})")
    sm = boto3.client("sagemaker")
    start = time.time()
    try:
        trainer.train(input_data_config=input_data_config, job_name=job_name)
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
    """Deploy a real-time endpoint and return (endpoint, endpoint_name)."""
    endpoint_name = random_suffix_name(f"xgb-{test_name}", 32)

    inference_spec = InferenceSpec(
        image_uri=image_uri,
        model_data_url=model_data,
    )
    if env:
        inference_spec.environment = env

    builder = ModelBuilder(
        inference_spec=inference_spec,
        role=role,
    )

    try:
        endpoint = builder.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
        )
    except Exception:
        raise

    _created_endpoints.append(endpoint_name)
    return endpoint, endpoint_name


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

    # Create the model via boto3 for batch transform
    sm = boto3.client("sagemaker")
    model_name = random_suffix_name(f"xgb-{test_name}-model", 32)
    create_model_params = {
        "ModelName": model_name,
        "PrimaryContainer": {
            "Image": image_uri,
            "ModelDataUrl": model_data,
        },
        "ExecutionRoleArn": role,
    }
    if env:
        create_model_params["PrimaryContainer"]["Environment"] = env
    sm.create_model(**create_model_params)
    _created_models.append(model_name)

    # Create and run transform job via boto3
    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3_uri,
                }
            },
            "ContentType": content_type,
            "SplitType": split_type,
        },
        TransformOutput={
            "S3OutputPath": output_path,
            "Accept": accept,
        },
        TransformResources={
            "InstanceType": instance_type,
            "InstanceCount": 1,
        },
    )

    # Wait for completion
    waiter = sm.get_waiter("transform_job_completed_or_stopped")
    try:
        waiter.wait(TransformJobName=job_name)
    except Exception:
        try:
            sm.stop_transform_job(TransformJobName=job_name)
        except Exception:
            pass
        raise

    desc = sm.describe_transform_job(TransformJobName=job_name)
    return desc
