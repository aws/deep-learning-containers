# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import re
import json
import logging
import requests
import argparse
import os
import sys


def _validate_instance_id(instance_id):
    """
    Validate instance ID
    """
    instance_id_regex = r"^(i-\S{17})"
    compiled_regex = re.compile(instance_id_regex)
    match = compiled_regex.match(instance_id)

    if not match:
        return None

    return match.group(1)


def _retrieve_instance_id():
    """
    Retrieve instance ID from instance metadata service
    """
    instance_id = None
    url = "http://169.254.169.254/latest/meta-data/instance-id"
    response = requests_helper(url, timeout=0.1)

    if response is not None and not (400 <= response.status_code < 600):
        instance_id = _validate_instance_id(response.text)

    return instance_id


def _retrieve_instance_region():
    """
    Retrieve instance region from instance metadata service
    """
    region = None
    valid_regions = [
        "ap-northeast-1",
        "ap-northeast-2",
        "ap-southeast-1",
        "ap-southeast-2",
        "ap-south-1",
        "ca-central-1",
        "eu-central-1",
        "eu-north-1",
        "eu-west-1",
        "eu-west-2",
        "eu-west-3",
        "sa-east-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ]

    url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
    response = requests_helper(url, timeout=0.1)

    if response is not None and not (400 <= response.status_code < 600):
        response_json = json.loads(response.text)

        if response_json["region"] in valid_regions:
            region = response_json["region"]

    return region


def parse_args():
    """
    Parsing function to parse input arguments.
    Return: args, which containers parsed input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework",
                        choices=["tensorflow", "mxnet", "pytorch"],
                        help="framework of container image.",
                        required=True)
    parser.add_argument("--framework-version",
                        help="framework version of container image.",
                        required=True)
    parser.add_argument("--container-type",
                        choices=["training", "inference"],
                        help="What kind of jobs you want to run on container. \
                        Either training or inference.",
                        required=True)

    args, _unknown = parser.parse_known_args()

    return args


def query_bucket():
    """
    GET request on an empty object from an Amazon S3 bucket
    """
    response = None
    instance_id = _retrieve_instance_id()
    region = _retrieve_instance_region()
    args = parse_args()
    framework, framework_version, container_type = args.framework, args.framework_version, args.container_type
    py_version = sys.version.split(" ")[0]

    if instance_id is not None and region is not None:
        url = (
            "https://aws-deep-learning-containers-{0}.s3.{0}.amazonaws.com"
            "/dlc-containers-{1}.txt?x-instance-id={1}&x-framework={2}&x-framework_version={3}&x-py_version={4}&x-container_type={5}".format(
                region, instance_id, framework, framework_version, py_version, container_type
            )
        )
        response = requests_helper(url, timeout=0.2)
        if os.environ.get("TEST_MODE") == str(1):
            with open(os.path.join(os.sep, "tmp", "test_request.txt"), "w+") as rf:
                rf.write(url)

    logging.debug("Query bucket finished: {}".format(response))

    return response


def requests_helper(url, timeout):
    response = None
    try:
        response = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        logging.error("Request exception: {}".format(e))

    return response


def main():
    """
    Invoke bucket query
    """
    # Logs are not necessary for normal run. Remove this line while debugging.
    logging.getLogger().disabled = True

    logging.basicConfig(level=logging.ERROR)
    query_bucket()


if __name__ == "__main__":
    main()
