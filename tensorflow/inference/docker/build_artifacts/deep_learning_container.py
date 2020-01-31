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


def _validate_instance_id(instance_id):
    """
    Validate instance ID
    """
    instance_id_regex = r'^(i-\S{17})'
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

    if response is not None:
        instance_id = _validate_instance_id(response.text)

    return instance_id


def _retrieve_instance_region():
    """
    Retrieve instance region from instance metadata service
    """
    region = None
    valid_regions = ['ap-northeast-1', 'ap-northeast-2', 'ap-southeast-1', 'ap-southeast-2',
                     'ap-south-1', 'ca-central-1', 'eu-central-1', 'eu-north-1',
                     'eu-west-1', 'eu-west-2', 'eu-west-3', 'sa-east-1',
                     'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2']

    url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
    response = requests_helper(url, timeout=0.1)

    if response is not None:
        response_json = json.loads(response.text)

        if response_json['region'] in valid_regions:
            region = response_json['region']

    return region


def query_bucket():
    """
    GET request on an empty object from an Amazon S3 bucket
    """
    response = None
    instance_id = _retrieve_instance_id()
    region = _retrieve_instance_region()

    if instance_id is not None and region is not None:
        url = ("https://aws-deep-learning-containers-{0}.s3.{0}.amazonaws.com"
               "/dlc-containers.txt?x-instance-id={1}".format(region, instance_id))
        response = requests_helper(url, timeout=0.2)

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


if __name__ == '__main__':
    main()
