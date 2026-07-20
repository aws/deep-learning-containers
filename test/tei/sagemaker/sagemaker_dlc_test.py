# Ported from awslabs/llm-hosting-container/tests/huggingface/sagemaker_dlc_test.py.
# Adjustments: env var names remapped to what our CI harness exports
# (IMAGE_URI -> TEST_IMAGE_URI, TEST_ROLE_ARN -> SM_ROLE_ARN, TARGET_IMAGE_TYPE hardcoded to TEI, DEVICE_TYPE -> TEST_DEVICE_TYPE).
import argparse
import json
import logging
import os
import signal
import sys
import time

import pytest
from sagemaker.huggingface import HuggingFaceModel

logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")


def run_test(args):
    default_env = {"HF_MODEL_ID": args.model_id}
    if args.model_revision:
        default_env["HF_MODEL_REVISION"] = args.model_revision
    if args.instance_type.startswith("ml.inf2"):
        default_env["MODEL_ID"] = default_env.pop("HF_MODEL_ID")
        default_env["HF_NUM_CORES"] = "2"
        default_env["HF_AUTO_CAST_TYPE"] = "bf16"
        default_env["MAX_BATCH_SIZE"] = "1"
        default_env["MAX_INPUT_TOKENS"] = "2048"
        default_env["MAX_TOTAL_TOKENS"] = "4096"
    else:
        default_env["SM_NUM_GPUS"] = "4"

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(args.timeout))
    predictor = None
    try:
        # Create Hugging Face Model Class
        endpoint_name = args.model_id.replace("/", "-").replace(".", "-")[:40]
        endpoint_name = endpoint_name + "-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        model = HuggingFaceModel(
            name=endpoint_name, env=default_env, role=args.role, image_uri=args.image_uri
        )
        deploy_parameters = {
            "instance_type": args.instance_type,
            "initial_instance_count": 1,
            "endpoint_name": endpoint_name,
            "container_startup_health_check_timeout": 1800,
        }
        if args.instance_type.startswith("ml.inf2"):
            deploy_parameters["volume_size"] = 256
            deploy_parameters["inference_ami_version"] = "al2-ami-sagemaker-inference-neuron-2"
        predictor = model.deploy(**deploy_parameters)

        logging.info("Endpoint deployment complete.")

        data = {
            "inputs": "What is Deep Learning?",
            "parameters": {"max_new_tokens": 50, "top_k": 50, "top_p": 0.95, "do_sample": True},
        }
        output = predictor.predict(data)
        logging.info("Output: " + json.dumps(output))
        # TODO: we need to clearly define the expected output format for each models.
        # assert "generated_text" in output[0]
    finally:
        if predictor:
            predictor.delete_model()
            predictor.delete_endpoint()
        signal.alarm(0)


def get_models_for_image(image_type, device_type):
    if image_type == "TGI":
        if device_type == "gpu":
            return [
                ("bigscience/bloom-560m", None, "ml.g6.12xlarge"),
                ("EleutherAI/gpt-neox-20b", None, "ml.g6.12xlarge"),
                ("google/flan-t5-xxl", None, "ml.g6.12xlarge"),
            ]
        elif device_type == "inf2":
            return [("Qwen/Qwen3-0.6B", None, "ml.inf2.8xlarge")]
        else:
            raise ValueError(
                f"No testing models found for {image_type} on instance {device_type}. "
                f"please check whether the image_type and instance_type are supported."
            )
    elif image_type == "TEI":
        if device_type == "gpu":
            return [
                ("BAAI/bge-m3", None, "ml.g4dn.2xlarge"),
                ("intfloat/multilingual-e5-base", None, "ml.g4dn.2xlarge"),
                ("thenlper/gte-base", None, "ml.g4dn.2xlarge"),
                ("sentence-transformers/all-MiniLM-L6-v2", None, "ml.g4dn.2xlarge"),
            ]
        elif device_type == "cpu":
            return [("BAAI/bge-m3", None, "ml.m5.xlarge")]
        else:
            raise ValueError(
                f"No testing models found for {image_type} on instance {device_type}. "
                f"please check whether the image_type and instance_type are supported."
            )
    else:
        raise ValueError("Invalid image type. Supported types are 'TGI' and 'TEI'.")


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
    arg_parser.add_argument("--instance_type", type=str, required=True)
    arg_parser.add_argument("--model_id", type=str, required=True)
    arg_parser.add_argument("--model_revision", type=str, required=False)
    arg_parser.add_argument("--role", type=str, required=True)
    arg_parser.add_argument("--timeout", type=str, required=True)

    args = arg_parser.parse_args()
    run_test(args)
