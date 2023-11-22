from __future__ import absolute_import

import os
import sys

from io import BytesIO
from PIL import Image
import pytest
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import BytesDeserializer

import time
import logging

from ...integration import sdxl_gpu_path, sd21_gpu_path, sdxl_gpu_code_path, sd21_gpu_code_path


from .timeout import timeout_and_delete_endpoint
from .... import invoke_pytorch_helper_function

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


@pytest.mark.model("sdxl")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.stabilityai_only
@pytest.mark.team("sagemaker-1p-algorithms")
def test_sdxl_v1_0_gpu_stabilityai(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = "ml.g5.4xlarge"
    model_bucket = "stabilityai-public-packages"
    sgm_version = "0.1.0"
    model_prefix = f"model-packages/sdxl-v1-0-dlc/sgm{sgm_version}"
    model_file = f"sdxlv1-sgm{sgm_version}.tar.gz"
    script_path = os.path.join(sdxl_gpu_code_path, f"sgm{sgm_version}", "inference.py")
    inference_request = {
        "text_prompts": [{"text": "A wonderous machine creating images"}],
        "height": 1024,
        "width": 1024,
    }
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_bucket": model_bucket,
        "model_prefix": model_prefix,
        "model_file": model_file,
        "inference_script": script_path,
        "inference_request": inference_request,
        "download_path": sdxl_gpu_path,
    }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_sgm_inference, function_args)


@pytest.mark.model("sd21")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.stabilityai_only
@pytest.mark.team("sagemaker-1p-algorithms")
def test_sd_v2_1_gpu_stabilityai(framework_version, ecr_image, instance_type, sagemaker_regions):
    instance_type = "ml.g5.xlarge"
    model_bucket = "stabilityai-public-packages"
    sgm_version = "0.1.0"
    model_prefix = f"model-packages/stable-diffusion-v2-1-dlc/sgm{sgm_version}"
    model_file = f"stable-diffusion-v2-1-sgm{sgm_version}.tar.gz"
    script_path = os.path.join(sd21_gpu_code_path, f"sgm{sgm_version}", "inference.py")
    inference_request = {
        "text_prompts": [{"text": "A wonderous machine creating images"}],
        "height": 512,
        "width": 512,
    }
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_bucket": model_bucket,
        "model_prefix": model_prefix,
        "model_file": model_file,
        "inference_script": script_path,
        "inference_request": inference_request,
        "download_path": sd21_gpu_path,
    }
    invoke_pytorch_helper_function(ecr_image, sagemaker_regions, _test_sgm_inference, function_args)


def _test_sgm_inference(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type,
    model_bucket,
    model_prefix,
    model_file,
    inference_script,
    inference_request,
    download_path,
    verify_logs=True,
):
    endpoint_name = sagemaker.utils.unique_name_from_base("sagemaker-stabilityai-pytorch-serving")

    model_data_file = os.path.join(download_path, model_file)
    if not os.path.exists(model_data_file):
        LOGGER.info(
            f"Downloading s3://{model_bucket}/{model_prefix}/{model_file} to {download_path}"
        )
        sagemaker_session.download_data(
            path=download_path, bucket=model_bucket, key_prefix=f"{model_prefix}/{model_file}"
        )
    else:
        LOGGER.info(f"Using existing file {model_data_file}")

    model_data = sagemaker_session.upload_data(
        path=model_data_file, key_prefix="sagemaker-stabilityai-pytorch-serving/models"
    )

    LOGGER.info(f"Using inference script {inference_script}")

    pytorch = PyTorchModel(
        model_data=model_data,
        role="SageMakerRole",
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        entry_point=inference_script,  # This seems to be ignored so the inference script built into the container is always run
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=60):
        predictor = pytorch.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=BytesDeserializer(accept="image/png"),
        )

        # Model loading can take up to 2 min so we must wait
        time.sleep(60 * 2)

        output = predictor.predict(inference_request)

        image = Image.open(BytesIO(output))
        assert image.height == inference_request["height"]
        assert image.width == inference_request["width"]

        #  Check for Cloudwatch logs
        if verify_logs:
            _check_for_cloudwatch_logs(endpoint_name, sagemaker_session)


def _check_for_cloudwatch_logs(endpoint_name, sagemaker_session):
    client = sagemaker_session.boto_session.client("logs")
    log_group_name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    time.sleep(30)
    identify_log_stream = client.describe_log_streams(
        logGroupName=log_group_name, orderBy="LastEventTime", descending=True, limit=5
    )

    try:
        log_stream_name = identify_log_stream["logStreams"][0]["logStreamName"]
    except IndexError as e:
        raise RuntimeError(
            f"Unable to look up log streams for the log group {log_group_name}"
        ) from e

    log_events_response = client.get_log_events(
        logGroupName=log_group_name, logStreamName=log_stream_name, limit=50, startFromHead=True
    )

    records_available = bool(log_events_response["events"])

    if not records_available:
        raise RuntimeError(
            f"records_available variable is false... No cloudwatch events getting logged for the group {log_group_name}"
        )
    else:
        LOGGER.info(
            f"Most recently logged events were found for the given log group {log_group_name} & log stream {log_stream_name}... Now verifying that TorchServe endpoint is logging on cloudwatch"
        )
        check_for_torchserve_response = client.filter_log_events(
            logGroupName=log_group_name,
            logStreamNames=[log_stream_name],
            filterPattern="Torch worker started.",
            limit=10,
            interleaved=False,
        )
        assert bool(check_for_torchserve_response["events"])
