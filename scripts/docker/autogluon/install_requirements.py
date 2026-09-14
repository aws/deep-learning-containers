#!/opt/venv/bin/python
"""Install model-artifact requirements once before Gunicorn starts."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlparse, urlunparse

LOGGER = logging.getLogger("autogluon-serving")


def _codeartifact_index(repository_arn: str) -> str:
    # Adapted from scripts/docker/ray/sagemaker_serve.py.
    match = re.fullmatch(
        r"arn:([^:]+):codeartifact:([^:]+):([^:]+):repository/([^/]+)/(.+)",
        repository_arn,
    )
    if not match:
        raise ValueError(f"Invalid CA_REPOSITORY_ARN: {repository_arn!r}")

    import boto3

    _, region, account, domain, repository = match.groups()
    client = boto3.client("codeartifact", region_name=region)
    arguments = {
        "domain": domain,
        "domainOwner": account,
    }
    token = client.get_authorization_token(**arguments)["authorizationToken"]
    endpoint = client.get_repository_endpoint(
        **arguments,
        repository=repository,
        format="pypi",
    )["repositoryEndpoint"]

    parsed = urlparse(endpoint)
    authenticated = parsed._replace(netloc=f"aws:{quote(token, safe='')}@{parsed.netloc}")
    return urlunparse(authenticated._replace(path=f"{parsed.path.rstrip('/')}/simple/"))


def install_requirements() -> None:
    requirements_path = (
        Path(os.getenv("SAGEMAKER_BASE_DIR", "/opt/ml")) / "model" / "code" / "requirements.txt"
    )
    if not requirements_path.is_file():
        return

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-cache",
        "-r",
        str(requirements_path),
    ]
    environment = os.environ.copy()
    repository_arn = os.getenv("CA_REPOSITORY_ARN")
    if repository_arn:
        environment["UV_INDEX_URL"] = _codeartifact_index(repository_arn)

    LOGGER.info("Installing packages from %s", requirements_path)
    subprocess.check_call(command, env=environment)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    install_requirements()


if __name__ == "__main__":
    main()
