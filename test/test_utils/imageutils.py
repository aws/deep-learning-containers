import re
import boto3
import json


def get_image_account_id(image_uri):
    """
    Find the account ID where the image is located
    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]


def get_image_repository_name(image_uri):
    repository_uri, _ = image_uri.split(":")
    _, repository = repository_uri.split("/")
    return repository


def get_image_tag_name(image_uri):
    _, tag = image_uri.split(":")
    return tag


def get_image_region(image_uri):
    """
    Find the region where the image is located
    :param image_uri: <str> ECR image URI
    :return: <str> AWS Region Name
    """
    region_pattern = r"(us(-gov)?|af|ap|ca|cn|eu|il|me|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()


def get_image_labels(image_uri, client=None):
    """
    Get all labels applied on the given image URI hosted on ECR through the image manifest.
    :param image_uri: str Input Image URI
    :param ecr_client: boto3 ECR Client object in the same region as the image URI
    :return: dict All Docker Image Labels applied on the image
    """
    account_id = get_image_account_id(image_uri)
    repo = get_image_repository_name(image_uri)
    tag = get_image_tag_name(image_uri)
    region = get_image_region(image_uri)
    if not client:
        client = boto3.client("ecr", region_name=region)

    labels = get_image_labels_with_manifest(client, repo, tag, account_id=account_id)
    return labels


def get_image_labels_with_manifest(client, repository, tag, account_id=None, manifest_kwargs=None):
    """
    Get all labels applied on an image hosted on ECR through the image manifest.
    :param ecr_client:
    :param repo_name:
    :param image_tag:
    :param account_id:
    :return: dict All Docker Image Labels applied on the image
    """
    if not manifest_kwargs:
        manifest_kwargs = {
            "acceptedMediaTypes": ["application/vnd.docker.distribution.manifest.v1+json"]
        }
    if account_id:
        manifest_kwargs["registryId"] = account_id

    manifest_str = get_image_manifest(
        repository=repository,
        tag=tag,
        client=client,
        **manifest_kwargs,
    )
    manifest = json.loads(manifest_str)
    metadata = json.loads(manifest["history"][0]["v1Compatibility"])
    labels = metadata["config"]["Labels"]
    return labels


def get_image_manifest(repository, tag, client, **kwargs):
    """
    Helper function to get an image manifest from ECR.
    :param image_repo: <str> Repository name
    :param image_tag: <str> Image tag to be queried
    :param ecr_client: <boto3.client> ECR client object to be used for query
    :return: ECR image manifest as dict, or requested format if mentioned in kwargs.
    """
    response = client.batch_get_image(
        repositoryName=repository, imageIds=[{"imageTag": tag}], **kwargs
    )
    if not response.get("images"):
        raise ValueError(
            f"Failed to get images through ecr_client.batch_get_image response for image {repository}:{tag}"
        )
    elif not response["images"][0].get("imageManifest"):
        raise KeyError(
            f"imageManifest not found in ecr_client.batch_get_image response:\n{response['images']}"
        )
    return response["images"][0]["imageManifest"]


def are_fixture_labels_enabled(image_uri, labels):
    """
    Returns False if a fixture label in the given image has value other than true
    Otherwise returns True
    Example:
    Expected fixture labels: [a,b,c]
    image labels: [] -> True
    image labels: [a=true] -> True # assumes [b=true, c=true]
    image labels: [a=true, b=false, c=true] -> False
    image labels: [a=true, b=true, c=true] -> True
    image labels: [a=false] -> False
    """
    image_labels = get_image_labels(image_uri)
    if image_labels:
        for label in labels:
            if image_labels.get(label, "True").lower() != "true":
                return False
    return True


def are_image_labels_matched(image_uri, labels):
    image_label_collection = get_image_labels(image_uri)
    if not image_label_collection:
        return False
    for label in labels:
        if label not in image_label_collection or image_label_collection[label].lower() != "true":
            return False
    return True
