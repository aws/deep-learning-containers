"""Shared fixtures and helpers for Scikit-learn SageMaker tests."""

import logging
import time

import boto3
import pytest
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.multidatamodel import MultiDataModel
from sagemaker.pipeline import PipelineModel
from test_utils import random_suffix_name

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

E2E_TEST_BUCKET = "amazonai-algorithms-integration-tests"
E2E_DATA_PREFIX = "input/scikit-learn"

_created_models = []
_created_endpoints = []


def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"


def data_uri(key):
    return s3_uri(E2E_TEST_BUCKET, f"{E2E_DATA_PREFIX}/{key}")


def cleanup_resources():
    """Delete all SageMaker resources created during the test session."""
    sm = boto3.client("sagemaker")
    LOGGER.info(
        f"Session cleanup: {len(_created_endpoints)} endpoints, {len(_created_models)} models"
    )
    for ep in _created_endpoints:
        try:
            sm.delete_endpoint(EndpointName=ep)
            LOGGER.info(f"Deleted endpoint {ep}")
        except Exception as e:
            LOGGER.warning(f"Failed to delete endpoint {ep}: {e}")
        try:
            sm.delete_endpoint_config(EndpointConfigName=ep)
            LOGGER.info(f"Deleted endpoint-config {ep}")
        except Exception as e:
            LOGGER.warning(f"Failed to delete endpoint-config {ep}: {e}")
    for model_name in _created_models:
        try:
            sm.delete_model(ModelName=model_name)
            LOGGER.info(f"Deleted model {model_name}")
        except Exception as e:
            LOGGER.warning(f"Failed to delete model {model_name}: {e}")
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
    content_type=None,
    test_name="train",
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size=10,
    max_run=1800,
    input_mode="File",
    train_distribution="ShardedByS3Key",
    enable_network_isolation=False,
    extra_channels=None,
):
    """Launch a SageMaker training job and return (job_name, duration, description)."""
    job_name = random_suffix_name(f"skl-{test_name}", 32)
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
        enable_network_isolation=enable_network_isolation,
    )

    train_input_kwargs = {"distribution": train_distribution}
    if content_type is not None:
        train_input_kwargs["content_type"] = content_type
    channels = {
        "train": TrainingInput(
            s3_data=data_uri(train_s3_key),
            **train_input_kwargs,
        ),
    }

    if extra_channels:
        for name, value in extra_channels.items():
            if isinstance(value, tuple):
                uri, ch_content_type = value
                channels[name] = TrainingInput(s3_data=uri, content_type=ch_content_type)
            else:
                channels[name] = TrainingInput(s3_data=value)

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
    """Deploy a real-time endpoint and return (predictor, endpoint_name)."""
    from sagemaker.predictor import Predictor

    endpoint_name = random_suffix_name(f"skl-{test_name}", 32)
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=role,
        env=env,
    )
    LOGGER.info(f"Deploying endpoint {endpoint_name} on {instance_type} (model_data={model_data})")
    start = time.time()
    try:
        model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
    except Exception:
        if model.name:
            _created_models.append(model.name)
        raise
    LOGGER.info(f"Endpoint {endpoint_name} InService in {time.time() - start:.0f}s")
    _created_models.append(model.name)
    _created_endpoints.append(endpoint_name)
    predictor = Predictor(endpoint_name=endpoint_name)
    return predictor, endpoint_name


def deploy_multi_model_endpoint(
    image_uri,
    role,
    model_data_prefix,
    test_name="mme",
    instance_type="ml.m5.xlarge",
    env=None,
):
    """Deploy a SageMaker Multi-Model Endpoint.

    `model_data_prefix` is the S3 prefix where model tarballs live
    (each tarball becomes a target model addressable by name).
    Returns (predictor, endpoint_name, mm_model).
    """
    from sagemaker.predictor import Predictor

    endpoint_name = random_suffix_name(f"skl-{test_name}", 32)
    mm_model = MultiDataModel(
        name=random_suffix_name(f"skl-{test_name}-model", 32),
        model_data_prefix=model_data_prefix,
        image_uri=image_uri,
        role=role,
        env=env,
    )
    LOGGER.info(f"Deploying MME {endpoint_name} on {instance_type} (prefix={model_data_prefix})")
    start = time.time()
    try:
        mm_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
    except Exception:
        if mm_model.name:
            _created_models.append(mm_model.name)
        raise
    LOGGER.info(f"MME {endpoint_name} InService in {time.time() - start:.0f}s")
    _created_models.append(mm_model.name)
    _created_endpoints.append(endpoint_name)
    predictor = Predictor(endpoint_name=endpoint_name)
    return predictor, endpoint_name, mm_model


def deploy_inference_pipeline(models, role, test_name="pipe", instance_type="ml.m5.xlarge"):
    """Deploy a multi-container inference pipeline endpoint.

    `models` is an ordered list of (image_uri, model_data, env) tuples.
    Returns (predictor, endpoint_name).
    """
    from sagemaker.predictor import Predictor

    endpoint_name = random_suffix_name(f"skl-{test_name}", 32)
    pipeline_models = [
        Model(image_uri=img, model_data=data, role=role, env=env) for img, data, env in models
    ]
    pipeline = PipelineModel(
        name=random_suffix_name(f"skl-{test_name}-model", 32),
        role=role,
        models=pipeline_models,
    )
    LOGGER.info(
        f"Deploying pipeline endpoint {endpoint_name} on {instance_type} "
        f"({len(pipeline_models)} containers)"
    )
    start = time.time()
    try:
        pipeline.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
        )
    except Exception:
        for m in pipeline_models:
            if m.name:
                _created_models.append(m.name)
        raise
    LOGGER.info(f"Pipeline endpoint {endpoint_name} InService in {time.time() - start:.0f}s")
    for m in pipeline_models:
        if m.name:
            _created_models.append(m.name)
    _created_endpoints.append(endpoint_name)
    predictor = Predictor(endpoint_name=endpoint_name)
    return predictor, endpoint_name


def predict_and_log(predictor, payload, **kwargs):
    """Invoke an endpoint and log latency + first 200 chars of the response."""
    payload_len = len(payload) if isinstance(payload, (str, bytes)) else len(str(payload))
    target_model = kwargs.get("target_model")
    target_suffix = f" target={target_model}" if target_model else ""
    LOGGER.info(f"POST {predictor.endpoint_name}{target_suffix} payload_len={payload_len}")
    start = time.time()
    response = predictor.predict(payload, **kwargs)
    LOGGER.info(
        f"{predictor.endpoint_name}{target_suffix} responded in "
        f"{(time.time() - start) * 1000:.0f}ms: {str(response)[:200]}"
    )
    return response


def delete_endpoint(endpoint_name):
    """Delete endpoint, endpoint config, and associated model(s)."""
    sm = boto3.client("sagemaker")
    try:
        ep_config = sm.describe_endpoint_config(EndpointConfigName=endpoint_name)
        for variant in ep_config.get("ProductionVariants", []):
            model_name = variant.get("ModelName")
            if model_name:
                try:
                    sm.delete_model(ModelName=model_name)
                    LOGGER.info(f"Deleted model {model_name}")
                except Exception as e:
                    LOGGER.warning(f"Failed to delete model {model_name}: {e}")
                if model_name in _created_models:
                    _created_models.remove(model_name)
    except Exception as e:
        LOGGER.warning(f"Failed to describe endpoint-config {endpoint_name}: {e}")
    try:
        sm.delete_endpoint(EndpointName=endpoint_name)
        LOGGER.info(f"Deleted endpoint {endpoint_name}")
    except Exception as e:
        LOGGER.warning(f"Failed to delete endpoint {endpoint_name}: {e}")
    try:
        sm.delete_endpoint_config(EndpointConfigName=endpoint_name)
        LOGGER.info(f"Deleted endpoint-config {endpoint_name}")
    except Exception as e:
        LOGGER.warning(f"Failed to delete endpoint-config {endpoint_name}: {e}")
    if endpoint_name in _created_endpoints:
        _created_endpoints.remove(endpoint_name)
