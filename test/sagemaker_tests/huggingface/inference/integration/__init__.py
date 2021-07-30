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
# TODO: current local test, tests without custom script
# mnist_cpu_script = oos.path.join(resources_path, 'tiny-distilbert-sst-2','inference.py')


ROLE = "dummy/unused-role"
DEFAULT_TIMEOUT = 20

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


class NoLogStreamFoundError(Exception):
    pass


class SageMakerEndpointFailure(Exception):
    pass


def get_logs_from_cloudwatch(e):
    # Check to see if we can get more information from CloudWatchLogs
    print(f"TESTING THAT THIS WORKS {e}")
    endpoint_regex = re.compile(r"Error hosting endpoint ((\w|-)+):")
    endpoint_match = endpoint_regex.search(str(e))
    if endpoint_match:
        logs_client = boto3.client('logs', region_name='us-west-2')
        endpoint = endpoint_match.group(1)
        log_stream_resp = logs_client.describe_log_streams(logGroupName=f"/aws/sagemaker/Endpoints/{endpoint}")
        all_traffic_log_stream = ""
        for log_stream in log_stream_resp.get('logStreams', []):
            log_stream_name = log_stream.get('logStreamName')
            # If we have AllTraffic log stream, just use that
            if log_stream_name.startswith("AllTraffic"):
                all_traffic_log_stream = log_stream_name
                break
        if not all_traffic_log_stream:
            raise NoLogStreamFoundError(f"Cannot find all traffic log streams for endpoint {endpoint}") from e
        raise SageMakerEndpointFailure(
            f"Error from endpoint {endpoint}:\n{json.dumps(all_traffic_log_stream, indent=4)}"
        ) from e
