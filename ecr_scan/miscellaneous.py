from docker import APIClient
from docker import DockerClient
import constants
import boto3
import botocore
import re
import json
import dataclasses
from datetime import date, datetime

def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data('endpoints'))

def push_image_to_ecr(image):
    #Check if the image is already present in the ecr repo only if not push the docker image
    docker_client = DockerClient(base_url=constants.DOCKER_URL)
    res = docker_client.images.push(image)

def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint('ecr', region)
    print("Inside ECR repo",endpoint_data)
    return '{}.dkr.{}'.format(account, endpoint_data['hostname'])

def get_unique_name_from_tag(image_uri):
    """
    Return the unique from the image tag.
    :param image_uri: ECR image URI
    :return: unique name
    """
    return re.sub("[^A-Za-z0-9]+", "", image_uri)

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    EnhancedJSONEncoder is required to dump dataclass objects as JSON.
    """

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)

