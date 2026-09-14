#!/opt/venv/bin/python
"""Install model-artifact requirements once before Gunicorn starts."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from urllib.parse import quote, urlsplit, urlunsplit

from settings import Settings

LOGGER = logging.getLogger("autogluon-serving")
CODEARTIFACT_ARN = re.compile(
    r"^arn:[^:]+:codeartifact:(?P<region>[^:]+):(?P<account>[^:]+):"
    r"repository/(?P<domain>[^/]+)/(?P<repository>[^/]+)$"
)


def _codeartifact_index(repository_arn: str) -> str:
    match = CODEARTIFACT_ARN.fullmatch(repository_arn)
    if match is None:
        raise ValueError(f"Invalid CA_REPOSITORY_ARN: {repository_arn!r}")

    import boto3

    values = match.groupdict()
    client = boto3.client("codeartifact", region_name=values["region"])
    arguments = {
        "domain": values["domain"],
        "domainOwner": values["account"],
    }
    token = client.get_authorization_token(**arguments)["authorizationToken"]
    endpoint = client.get_repository_endpoint(
        **arguments,
        repository=values["repository"],
        format="pypi",
    )["repositoryEndpoint"]

    parsed = urlsplit(endpoint)
    path = f"{parsed.path.rstrip('/')}/simple/"
    authenticated_host = f"aws:{quote(token, safe='')}@{parsed.netloc}"
    return urlunsplit((parsed.scheme, authenticated_host, path, parsed.query, parsed.fragment))


def install_requirements(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_environment()
    if not settings.requirements_path.is_file():
        return

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-cache",
        "-r",
        str(settings.requirements_path),
    ]
    environment = os.environ.copy()
    if settings.codeartifact_repository_arn:
        environment["UV_INDEX_URL"] = _codeartifact_index(settings.codeartifact_repository_arn)

    LOGGER.info("Installing packages from %s", settings.requirements_path)
    subprocess.check_call(command, env=environment)


def main() -> None:
    settings = Settings.from_environment()
    logging.basicConfig(level=settings.log_level)
    install_requirements(settings)


if __name__ == "__main__":
    main()
