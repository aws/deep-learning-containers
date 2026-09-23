"""SageMaker endpoint integration tests for TEI using SageMaker SDK v3."""

import argparse
import json
import logging
import os
import signal
import sys

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import random_suffix_name
from test_utils.instance_capacity import deploy_with_capacity_fallback

logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

MODEL_S3_PREFIX = "s3://dlc-cicd-models/tei-models"
INFERENCE_AMI_VERSION_CU12 = "al2-ami-sagemaker-inference-gpu-3-1"
# Keep the existing size first; all candidates have one NVIDIA L4 GPU.
GPU_INSTANCE_TYPES = ["ml.g6.4xlarge", "ml.g6.2xlarge", "ml.g6.xlarge"]


def model_data_uri(model_id):
    slug = model_id.replace("/", "__")
    return f"{MODEL_S3_PREFIX}/{slug}.tar.gz"


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")


def _cleanup(resources):
    """Delete all created resources, including after a partial deployment failure."""
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as error:
            logging.warning("Cleanup %s failed: %s", type(resource).__name__, error)


def _deploy_endpoint(args, instance_type):
    """Deploy one candidate, cleaning up partial resources before a retry."""
    default_env = {"HF_MODEL_ID": "/opt/ml/model"}
    if args.model_revision:
        default_env["HF_MODEL_REVISION"] = args.model_revision

    model = endpoint_config = endpoint = None
    try:
        model_slug = args.model_id.replace("/", "-").replace(".", "-")[:40]
        endpoint_name = random_suffix_name(f"tei-{model_slug}", 63)
        logging.info("Deploying %s on %s", endpoint_name, instance_type)
        model = Model.create(
            model_name=endpoint_name,
            primary_container=ContainerDefinition(
                image=args.image_uri,
                model_data_url=model_data_uri(args.model_id),
                environment=default_env,
            ),
            execution_role_arn=args.role,
        )
        variant_parameters = {
            "variant_name": "AllTraffic",
            "model_name": endpoint_name,
            "instance_type": instance_type,
            "initial_instance_count": 1,
            "container_startup_health_check_timeout_in_seconds": 1800,
        }
        if instance_type.startswith("ml.g") or instance_type.startswith("ml.p"):
            variant_parameters["inference_ami_version"] = INFERENCE_AMI_VERSION_CU12
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[ProductionVariant(**variant_parameters)],
        )
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        endpoint.wait_for_status("InService", timeout=int(args.timeout))
    except Exception:
        _cleanup([endpoint, endpoint_config, model])
        raise

    return model, endpoint_config, endpoint


def run_test(args):
    # The timeout covers all deployment attempts and inference for this model.
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(args.timeout))
    model = endpoint_config = endpoint = None
    try:
        model, endpoint_config, endpoint = deploy_with_capacity_fallback(
            args.instance_type,
            lambda candidate: _deploy_endpoint(args, candidate),
            args.model_id,
        )

        logging.info("Endpoint deployment complete.")

        data = {"inputs": "What is Deep Learning?"}
        result = endpoint.invoke(body=json.dumps(data), content_type="application/json")
        output = json.loads(result.body.read())
        logging.info("Output: " + json.dumps(output))
        assert output, "Model response is empty, failing endpoint test!"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        _cleanup([endpoint, endpoint_config, model])


def get_models_for_image(image_type, device_type):
    if image_type == "TEI":
        if device_type == "gpu":
            return [
                ("BAAI/bge-m3", None, GPU_INSTANCE_TYPES),
                ("intfloat/multilingual-e5-base", None, GPU_INSTANCE_TYPES),
                ("thenlper/gte-base", None, GPU_INSTANCE_TYPES),
                ("sentence-transformers/all-MiniLM-L6-v2", None, GPU_INSTANCE_TYPES),
            ]
        elif device_type == "cpu":
            return [("BAAI/bge-m3", None, "ml.m5.xlarge")]
        else:
            raise ValueError(
                f"No testing models found for {image_type} on instance {device_type}. "
                f"please check whether the image_type and instance_type are supported."
            )
    else:
        raise ValueError("Invalid image type. Supported type is 'TEI'.")


def should_run_test_for_image(test_type, target_type):
    return test_type == target_type


@pytest.mark.parametrize(
    "image_type, device_type",
    [
        pytest.param("TEI", "gpu", marks=pytest.mark.gpu),
        pytest.param("TEI", "cpu", marks=pytest.mark.cpu),
    ],
)
def test(image_type, device_type, timeout: str = "3000"):
    # Multi-image gating preserved from upstream; TEI is currently the only image.
    test_target_image_type = "TEI"
    test_device_type = os.getenv("TEST_DEVICE_TYPE")
    if test_target_image_type and not should_run_test_for_image(image_type, test_target_image_type):
        pytest.skip(
            f"Skipping test for image type {image_type} as it does not match target image type {test_target_image_type}"
        )

    if test_device_type and not should_run_test_for_image(device_type, test_device_type):
        pytest.skip(
            f"Skipping test for device type {device_type} as it does not match current device type {test_device_type}"
        )

    image_uri = os.getenv("TEST_IMAGE_URI")
    test_role_arn = os.getenv("SM_ROLE_ARN")
    assert image_uri, "Please set TEST_IMAGE_URI environment variable."
    assert test_role_arn, "Please set SM_ROLE_ARN environment variable."

    models = get_models_for_image(image_type, device_type)
    for model_id, model_revision, instance_type in models:
        args = argparse.Namespace(
            image_uri=image_uri,
            instance_type=instance_type,
            model_id=model_id,
            model_revision=model_revision,
            role=test_role_arn,
            timeout=timeout,
        )
        logging.info(f"Running sanity test with the following args: {args}.")
        run_test(args)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--image_uri", type=str, required=True)
    arg_parser.add_argument(
        "--instance_type",
        type=str,
        nargs="+",
        required=True,
        help="Instance types in fallback order",
    )
    arg_parser.add_argument("--model_id", type=str, required=True)
    arg_parser.add_argument("--model_revision", type=str, required=False)
    arg_parser.add_argument("--role", type=str, required=True)
    arg_parser.add_argument("--timeout", type=str, required=True)

    args = arg_parser.parse_args()
    run_test(args)
