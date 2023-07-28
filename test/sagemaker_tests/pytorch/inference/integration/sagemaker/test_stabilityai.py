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

import boto3
from datetime import datetime, timedelta
import time
import json
import logging

from ...integration import (
    sdxl_gpu_path,
    sdxl_gpu_script
)


from .timeout import timeout_and_delete_endpoint
from .... import invoke_pytorch_helper_function

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


@pytest.mark.model("sdxl")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
@pytest.mark.stabilityai_only
def test_sdxl_v1_0_gpu_stabilityai(
    framework_version, ecr_image, instance_type, sagemaker_regions
):
    instance_type = instance_type or "ml.g5.4xlarge"
    model_bucket = "stabilityai-public-packages"
    model_prefix = "model-packages/sdxl-v1-0-dlc"
    model_file = "model.tar.gz"
    inference_request = {
        "text_prompts": [
            {"text": "A wonderous machine creating images"}
        ],
        "height": 1024,
        "width": 1024
    }
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "model_bucket": model_bucket,
        "model_prefix": model_prefix,
        "model_file": model_file,
        "sdxl_script": sdxl_gpu_script,
        "inference_request": inference_request
    }
    invoke_pytorch_helper_function(
        ecr_image, sagemaker_regions, _test_sdxl_v1_0, function_args
    )


def _test_sdxl_v1_0(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type,
    model_bucket,
    model_prefix,
    model_file,
    sdxl_script,
    inference_request,
    verify_logs=True,
):
    endpoint_name = sagemaker.utils.unique_name_from_base(
        "sagemaker-pytorch-serving")

    LOGGER.info(
        f'Downloading s3://{model_bucket}{model_prefix} to {sdxl_gpu_path}')
    sagemaker_session.download_data(
        path=sdxl_gpu_path, bucket=model_bucket, key_prefix=f'{model_prefix}/{model_file}')
    
    model_data = sagemaker_session.upload_data(
        path=os.path.join(sdxl_gpu_path, model_file), key_prefix="sagemaker-pytorch-serving/models")

    pytorch = PyTorchModel(
        model_data=model_data,
        role="SageMakerRole",
        framework_version=framework_version,
        image_uri=ecr_image,
        sagemaker_session=sagemaker_session,
        entry_point=sdxl_script,
        env={
            # TODO bake these into the container?
            "TS_DEFAULT_RESPONSE_TIMEOUT": "1000",
            "HUGGINGFACE_HUB_CACHE": "/tmp/cache/huggingface/hub"
        }
    )

    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=30):
        predictor = pytorch.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=BytesDeserializer(accept="image/png")
        )

        # Model loading can take up to 10 min so we must wait
        time.sleep(60*10)
        
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
