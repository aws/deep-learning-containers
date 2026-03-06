"""Docker helper utilities for inspecting images and parsing URIs."""

import json
import subprocess
from typing import NamedTuple


class ImageURI(NamedTuple):
    full_uri: str
    account_id: str
    region: str
    repository: str
    image_tag: str


def parse_image_uri(image_uri: str) -> ImageURI:
    # Expected format: <account_id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
    registry, repository_tag = image_uri.split("/", 1)
    repository, image_tag = repository_tag.rsplit(":", 1)
    registry_parts = registry.split(".")
    account_id = registry_parts[0]
    region = registry_parts[3]
    return ImageURI(
        full_uri=image_uri,
        account_id=account_id,
        region=region,
        repository=repository,
        image_tag=image_tag,
    )


def get_docker_labels(image_uri: str) -> dict:
    """Get labels from a local Docker image via docker inspect."""
    result = subprocess.run(
        ["docker", "inspect", "--format={{json .Config.Labels}}", image_uri],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip()) or {}
