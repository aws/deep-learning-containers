# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import re
import json

import boto3

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


model_dir = os.path.join(resources_path, "tiny-distilbert-sst-2")
pt_model = "pt_model.tar.gz"
tf_model = "tf_model.tar.gz"
pt_neuron_model = "pt_neuron_model.tar.gz"
pt_neuronx_model = "pt_neuronx_model.tar.gz"
# TODO: current local test, tests without custom script

script_dir = os.path.join(resources_path, "code")
pt_neuron_script = "neuron_inference.py"
pt_neuronx_script = "neuronx_inference.py"
pt_ipex_script = "ipex_inference.py"

audio_sample_file_path = os.path.join(resources_path, "audio", "sample1.wav")
image_sample_file_path = os.path.join(resources_path, "image", "tiger.png")


ROLE = "dummy/unused-role"
DEFAULT_TIMEOUT = 20

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


class NoLogStreamFoundError(Exception):
    pass


class SageMakerEndpointFailure(Exception):
    pass


def dump_logs_from_cloudwatch(e, region="us-west-2"):
    """
    Function to dump logs from cloudwatch during error handling
    """
    error_hosting_endpoint_regex = re.compile(r"Error hosting endpoint ((\w|-)+):")
    endpoint_url_regex = re.compile(r"/aws/sagemaker/Endpoints/((\w|-)+)")
    endpoint_match = error_hosting_endpoint_regex.search(str(e)) or endpoint_url_regex.search(
        str(e)
    )
    if endpoint_match:
        logs_client = boto3.client("logs", region_name=region)
        endpoint = endpoint_match.group(1)
        log_group_name = f"/aws/sagemaker/Endpoints/{endpoint}"
        log_stream_resp = logs_client.describe_log_streams(logGroupName=log_group_name)
        all_traffic_log_stream = ""
        for log_stream in log_stream_resp.get("logStreams", []):
            log_stream_name = log_stream.get("logStreamName")
            # Format of AllTraffic log stream should be AllTraffic/<instance_id>
            if log_stream_name.startswith("AllTraffic"):
                all_traffic_log_stream = log_stream_name
                break
        if not all_traffic_log_stream:
            raise NoLogStreamFoundError(
                f"Cannot find all traffic log streams for endpoint {endpoint}"
            ) from e
        events = logs_client.get_log_events(
            logGroupName=log_group_name, logStreamName=all_traffic_log_stream
        )
        raise SageMakerEndpointFailure(
            f"Error from endpoint {endpoint}:\n{json.dumps(events, indent=4)}"
        ) from e


def invoke_sm_helper_function(
    ecr_image, sagemaker_regions, helper_function, helper_function_args
):
    """
    Used to invoke SM job defined in the helper functions in respective test file. The ECR image and the sagemaker
    session are passed explicitly depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues are seen.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param helper_function: Function to invoke
    :param helper_function_args: Helper function args

    :return: None
    """
    from .... import get_ecr_image_region, get_sagemaker_session, get_ecr_image
    import sagemaker

    ecr_image_region = get_ecr_image_region(ecr_image)
    for i, region in enumerate(sagemaker_regions):
        sagemaker_session = get_sagemaker_session(region)
        # Reupload the image to test region if needed
        tested_ecr_image = (
            get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        )
        try:
            helper_function(tested_ecr_image, sagemaker_session, **helper_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e) and i != len(sagemaker_regions) - 1:
                continue
            else:
                dump_logs_from_cloudwatch(e, region)
                raise e
        except Exception as e:
            dump_logs_from_cloudwatch(e, region)
            raise e
