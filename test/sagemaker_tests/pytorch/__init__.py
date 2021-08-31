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
import os
import sagemaker
import boto3
import subprocess
import re
import time
from sagemaker import Session
from base64 import b64decode


def get_sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.

    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub('[^A-Za-z0-9]+', '', image_uri)


def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]


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


def delete_file(file_path):
    subprocess.check_output(f"rm -rf {file_path}", shell=True, executable="/bin/bash")


def get_ecr_login_boto3(ecr_client, account_id, region):
    """
    Get ECR login using boto3
    """
    user_name, password = None, None
    result = ecr_client.get_authorization_token()
    for auth in result['authorizationData']:
        auth_token = b64decode(auth['authorizationToken']).decode()
        user_name, password = auth_token.split(':')
    return user_name, password


def save_credentials_to_file(file_path, password):
    with open(file_path, "w") as file:
        file.write(f"{password}")


class ECRRepoDoesNotExist(Exception):
    pass


def reupload_image_to_test_ecr(source_image_uri, target_image_repo_name, target_region):
    """
    Helper function to reupload an image owned by a another/same account to an ECR repo in this account to given region, so that
    this account can freely run tests without permission issues.

    :param source_image_uri: str Image URI for image to be tested
    :param target_image_repo_name: str Target image ECR repo name
    :param target_region: str Region where test is being run
    :return: str New image URI for re-uploaded image
    """
    ECR_PASSWORD_FILE_PATH = os.path.join("/tmp", f"{get_unique_name_from_tag(source_image_uri)}.txt")
    sts_client = boto3.client("sts", region_name=target_region)
    target_ecr_client = boto3.client("ecr", region_name=target_region)
    target_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(source_image_uri)
    image_region = get_region_from_image_uri(source_image_uri)
    image_repo_uri, image_tag = source_image_uri.split(":")
    _, image_repo_name = image_repo_uri.split("/")
    if not ecr_repo_exists(target_ecr_client, target_image_repo_name):
        raise ECRRepoDoesNotExist(
            f"Repo named {target_image_repo_name} does not exist in {target_region} on the account {target_account_id}"
        )

    target_image_uri = (
        source_image_uri.replace(image_region, target_region)
        .replace(image_repo_name, target_image_repo_name)
        .replace(image_account_id, target_account_id)
    )

    client = boto3.client('ecr', region_name = image_region)
    username, password = get_ecr_login_boto3(client, image_account_id, image_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)

    # using ctx.run throws error on codebuild "OSError: reading from stdin while output is captured".
    # Also it throws more errors related to awscli if in_stream=False flag is added to ctx.run which needs more deep dive
    subprocess.check_output(f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{image_account_id}.dkr.ecr.{image_region}.amazonaws.com && docker pull {source_image_uri}", shell=True, executable="/bin/bash")
    subprocess.check_output(f"docker tag {source_image_uri} {target_image_uri}", shell=True, executable="/bin/bash")
    delete_file(ECR_PASSWORD_FILE_PATH)
    username, password = get_ecr_login_boto3(target_ecr_client, target_account_id, target_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)
    subprocess.check_output(f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{target_account_id}.dkr.ecr.{target_region}.amazonaws.com && docker push {target_image_uri}", shell=True, executable="/bin/bash")
    delete_file(ECR_PASSWORD_FILE_PATH)

    return target_image_uri


def get_ecr_image_region(ecr_image):
    ecr_registry, _ = ecr_image.split("/")
    region_search = re.search(r"(us(-gov)?|ap|ca|cn|eu|sa|af|me)-(central|(north|south)?(east|west)?)-\d+", ecr_registry)
    return region_search.group()


def get_ecr_image(ecr_image, region):
    """
    It uploads image to the aws region and return image uri
    """
    image_repo_uri, image_tag = ecr_image.split(":")
    _, image_repo_name = image_repo_uri.split("/")
    target_image_repo_name = f"{image_repo_name}"
    regional_ecr_image = reupload_image_to_test_ecr(ecr_image, target_image_repo_name, region)
    return regional_ecr_image


def invoke_pytorch_helper_function(ecr_image, sagemaker_regions, helper_function, helper_function_args):
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

    num_retries = 2
    retry_delay = 300
    ecr_image_region = get_ecr_image_region(ecr_image)
    for _ in range(num_retries):
        for region in sagemaker_regions:
            sagemaker_session = get_sagemaker_session(region)
            # Reupload the image to test region if needed
            tested_ecr_image = get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
            try:
                helper_function(tested_ecr_image, sagemaker_session, **helper_function_args)
                return
            except sagemaker.exceptions.UnexpectedStatusException as e:
                if "CapacityError" in str(e):
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e
