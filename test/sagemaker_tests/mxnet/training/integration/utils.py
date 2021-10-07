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
import random
import time
import re
import boto3
import pytest
import subprocess
from base64 import b64decode
import sys
import botocore


def unique_name_from_base(base, max_length=63):
    unique = '%04x' % random.randrange(16**4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return '{}-{}-{}'.format(trimmed, ts, unique)


def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data('endpoints'))


def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint('ecr', region)
    return '{}.dkr.{}'.format(account, endpoint_data['hostname'])


def get_repository_and_tag_from_image_uri(image_uri):
    """
    Return the name of the repository holding the image

    :param image_uri: URI of the image
    :return: <str> repository name
    """
    repository_uri, tag = image_uri.split(":")
    _, repository_name = repository_uri.split("/")
    return repository_name, tag


def get_region_from_image_uri(image_uri):
    """
    Find the region where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Region Name
    """
    region_pattern = r"(us(-gov)?|ap|ca|cn|eu|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()


def ecr_repo_exists(ecr_client, repo_name, account_id=None):
    """
    :param ecr_client: boto3.Client for ECR
    :param repo_name: str ECR Repository Name
    :param account_id: str Account ID where repo is expected to exist
    :return: bool True if repo exists, False if not
    """
    query = {"repositoryNames": [repo_name]}
    if account_id:
        query["registryId"] = account_id
    try:
        ecr_client.describe_repositories(**query)
    except ecr_client.exceptions.RepositoryNotFoundException as e:
        return False
    return True


def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]
    

def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.

    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub('[^A-Za-z0-9]+', '', image_uri)


def reupload_image_to_test_ecr(source_image_uri, target_image_repo_name, target_region):
    """
    Helper function to reupload an image owned by a another/same account to an ECR repo in this account to given region, so that
    this account can freely run tests without permission issues.

    :param source_image_uri: str Image URI for image to be tested
    :param target_image_repo_name: str Target image ECR repo name
    :param target_region: str Region where test is being run
    :return: str New image URI for re-uploaded image
    """
    ecr_handler = EcrHandler()
    sts_client = boto3.client("sts", region_name=target_region)
    target_ecr_client = boto3.client("ecr", region_name=target_region)
    target_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(source_image_uri)
    image_region = get_region_from_image_uri(source_image_uri)
    image_repo_uri, image_tag = source_image_uri.split(":")
    _, image_repo_name = image_repo_uri.split("/")
    client = boto3.client('ecr', region_name = image_region)
    if not ecr_repo_exists(target_ecr_client, target_image_repo_name):
        raise ECRRepoDoesNotExist(
            f"Repo named {target_image_repo_name} does not exist in {target_region} on the account {target_account_id}"
        )

    target_image_uri = (
        source_image_uri.replace(image_region, target_region)
        .replace(image_repo_name, target_image_repo_name)
        .replace(image_account_id, target_account_id)
    )


    # using ctx.run throws error on codebuild "OSError: reading from stdin while output is captured".
    # Also it throws more errors related to awscli if in_stream=False flag is added to ctx.run which needs more deep dive
    ecr_handler.login_into_ecr(client, image_account_id)
    subprocess.check_output(f"docker pull {source_image_uri} && docker tag {source_image_uri} {target_image_uri}", shell=True, executable="/bin/bash")

    ecr_handler.login_into_ecr(target_ecr_client, target_account_id)
    subprocess.check_output(f"docker push {target_image_uri}", shell=True, executable="/bin/bash")

    return target_image_uri
