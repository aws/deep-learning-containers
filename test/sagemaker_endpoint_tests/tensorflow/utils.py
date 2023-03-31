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
import contextlib
import json
import boto3
import logging
import os
import sagemaker
import botocore

from sagemaker import utils

LOGGER = logging.getLogger(__name__)


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
    return f'{account}.dkr.{endpoint_data["hostname"]}'


def get_sagemaker_session(region, default_bucket=None):
    return sagemaker.Session(boto_session=boto3.Session(region_name=region), default_bucket=default_bucket)


def image_uri(registry, region, repo, tag):
    ecr_registry = get_ecr_registry(registry, region)
    return f"{ecr_registry}/{repo}:{tag}"


def _execution_role(boto_session):
    return boto_session.resource("iam").Role("SageMakerRole").arn

def create_endpoint_name(prefix):
    return utils.unique_name_from_base(prefix)

@contextlib.contextmanager
def sagemaker_model(boto_session, sagemaker_client, image_uri, model_name, model_data):
    container = {
        "Image": image_uri,
        "ModelDataUrl": model_data
    }
    model = sagemaker_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=_execution_role(boto_session),
        PrimaryContainer=container
    )
    try:
        yield model
    finally:
        LOGGER.info("deleting model %s", model_name)
        sagemaker_client.delete_model(ModelName=model_name)


def _test_bucket(region, boto_session):
    domain_suffix = ".cn" if region in ("cn-north-1", "cn-northwest-1") else ""
    sts_regional_endpoint = "https://sts.{}.amazonaws.com{}".format(region, domain_suffix)
    sts = boto_session.client(
        "sts",
        region_name=region,
        endpoint_url=sts_regional_endpoint
    )
    account = sts.get_caller_identity()["Account"]
    return f"sagemaker-{region}-{account}"


def find_or_put_model_data(region, boto_session, local_path):
    model_file = os.path.basename(local_path)

    bucket = _test_bucket(region, boto_session)
    key = "test-tfs/{}".format(model_file)

    s3 = boto_session.client("s3", region)

    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

        # bucket doesn't exist, create it
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={"LocationConstraint": region})

    try:
        s3.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] != "404":
            raise

        # file doesn't exist - upload it
        s3.upload_file(local_path, bucket, key)

    return "s3://{}/{}".format(bucket, key)
    