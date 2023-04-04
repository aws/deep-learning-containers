import os
import boto3
import botocore
import sagemaker
from sagemaker import utils

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

def create_endpoint_name(prefix):
    return utils.unique_name_from_base(prefix)
