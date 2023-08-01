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

import re
import time
import logging
import sys

import botocore

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data("endpoints"))


def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint("ecr", region)
    return "{}.dkr.{}".format(account, endpoint_data["hostname"])


def check_for_cloudwatch_logs(endpoint_name, sagemaker_session):
    client = sagemaker_session.boto_session.client("logs")
    log_group_name = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    time.sleep(30)
    identify_log_stream = client.describe_log_streams(
        logGroupName=log_group_name, orderBy="LastEventTime", descending=True, limit=5
    )

    all_traffic_log_stream = ""
    logs = identify_log_stream["logStreams"]

    while logs and not all_traffic_log_stream:
        log = logs.pop(0)
        if re.search(r"^AllTraffic/i-\w+$", log["logStreamName"]):
            all_traffic_log_stream = log["logStreamName"]
    if not all_traffic_log_stream:
        raise RuntimeError(
            f"Unable to look up log streams for the log group {log_group_name}"
        )

    log_events_response = client.get_log_events(
        logGroupName=log_group_name, logStreamName=all_traffic_log_stream, limit=50, startFromHead=True
    )

    records_available = bool(log_events_response["events"])

    if not records_available:
        raise RuntimeError(
            f"records_available variable is false... No cloudwatch events getting logged for the group {log_group_name}"
        )
    else:
        LOGGER.info(
            f"Most recently logged events were found for the given log group {log_group_name} & log stream {all_traffic_log_stream}... Now verifying that TorchServe endpoint is logging on cloudwatch"
        )
        check_for_torchserve_response = client.filter_log_events(
            logGroupName=log_group_name,
            logStreamNames=[all_traffic_log_stream],
            filterPattern="Torch worker started.",
            limit=10,
            interleaved=False,
        )
        assert bool(check_for_torchserve_response["events"])
