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

import json
import os
import re
import shutil
import tarfile

import boto3

# Path to test resources
resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))

# Model artifacts for local mode tests - downloaded from HuggingFace Hub at runtime
MODEL_ID = "Qwen/Qwen2.5-0.5B"
model_dir = os.path.join(resources_path, "qwen2.5-0.5b")
model_data = "qwen2.5-0.5b.tar.gz"
model_data_path = os.path.join(model_dir, model_data)


def ensure_model_downloaded():
    """Download model from HuggingFace Hub and create tarball if not already present."""
    if os.path.exists(model_data_path):
        return model_data_path

    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)
    local_model_dir = os.path.join(model_dir, "model")

    print(f"Downloading {MODEL_ID} from HuggingFace Hub...")
    snapshot_download(
        repo_id=MODEL_ID, local_dir=local_model_dir, ignore_patterns=["*.gguf", "*.onnx"]
    )

    # Remove cache folder if present
    cache_dir = os.path.join(local_model_dir, ".cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print(f"Creating tarball {model_data}...")
    with tarfile.open(model_data_path, "w:gz") as tar:
        for item in os.listdir(local_model_dir):
            tar.add(os.path.join(local_model_dir, item), arcname=item)

    # Clean up extracted model
    shutil.rmtree(local_model_dir)

    print(f"Model ready at {model_data_path}")
    return model_data_path


# Role for local mode (not used but required by SageMaker SDK)
ROLE = "dummy/unused-role"
DEFAULT_TIMEOUT = 45


class NoLogStreamFoundError(Exception):
    pass


class SageMakerEndpointFailure(Exception):
    pass


def dump_logs_from_cloudwatch(e, region="us-west-2"):
    """
    Function to dump logs from cloudwatch during error handling.
    Gracefully handles missing log groups/streams.
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
        try:
            log_stream_resp = logs_client.describe_log_streams(logGroupName=log_group_name)
            all_traffic_log_stream = ""
            for log_stream in log_stream_resp.get("logStreams", []):
                log_stream_name = log_stream.get("logStreamName")
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
        except logs_client.exceptions.ResourceNotFoundException:
            # Log group doesn't exist yet - endpoint may have failed before creating logs
            raise SageMakerEndpointFailure(
                f"Endpoint {endpoint} failed. No CloudWatch logs available yet."
            ) from e
