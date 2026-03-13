"""
SageMaker Ray Serve Adapter

Provides SageMaker-compatible endpoints (/invocations, /ping) that proxy to Ray Serve.
Ray Serve runs on port 8000 (internal), this adapter runs on port 8080 (SageMaker requirement).

Includes CodeArtifact support for runtime requirements.txt installation.
"""

import json
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from urllib.parse import urlparse, urlunparse

import boto3
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ray Serve backend URL (internal)
RAYSERVE_URL = os.getenv("RAYSERVE_BACKEND_URL", "http://127.0.0.1:8000")
REQUIREMENTS_PATH = "/opt/ml/model/code/requirements.txt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage HTTP client lifecycle."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        app.state.client = client
        yield


app = FastAPI(lifespan=lifespan)


def install_requirements():
    """Install packages from requirements.txt with optional CodeArtifact support."""
    if not os.path.exists(REQUIREMENTS_PATH):
        logger.info(f"No requirements.txt found at {REQUIREMENTS_PATH}, skipping")
        return

    logger.info(f"Installing packages from {REQUIREMENTS_PATH}...")

    pip_cmd = ["pip", "install", "-r", REQUIREMENTS_PATH]

    # Add CodeArtifact index if CA_REPOSITORY_ARN is set, if CA is configured but fails, hard fail
    ca_arn = os.getenv("CA_REPOSITORY_ARN")
    if ca_arn:
        ca_index = get_codeartifact_index(ca_arn)
        pip_cmd.extend(["--index-url", ca_index])

    try:
        subprocess.check_call(pip_cmd)
        logger.info("Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        sys.exit(1)


def get_codeartifact_index(repository_arn):
    """
    Build authenticated CodeArtifact index URL from CA_REPOSITORY_ARN.

    Uses boto3 APIs to get endpoint and auth token dynamically.

    Args:
        repository_arn: ARN like arn:aws:codeartifact:region:account:repository/domain/repo

    Returns:
        str: Authenticated pip index URL, or None if failed
    """
    # Parse ARN components
    arn_pattern = r"arn:([^:]+):codeartifact:([^:]+):([^:]+):repository/([^/]+)/(.+)"
    match = re.match(arn_pattern, repository_arn)
    if not match:
        raise ValueError(f"Invalid CA_REPOSITORY_ARN: {repository_arn}")

    _, region, account, domain, repository = match.groups()
    logger.info(f"Using CodeArtifact: {domain}/{repository} in {region}")

    try:
        ca = boto3.client("codeartifact", region_name=region)

        # Get auth token (12 hour expiry)
        token = ca.get_authorization_token(
            domain=domain, domainOwner=account, durationSeconds=43200
        )["authorizationToken"]

        # Get repository endpoint URL
        endpoint = ca.get_repository_endpoint(
            domain=domain, domainOwner=account, repository=repository, format="pypi"
        )["repositoryEndpoint"]

        # Parse URL and inject auth credentials
        parsed = urlparse(endpoint)
        authenticated = parsed._replace(netloc=f"aws:{token}@{parsed.netloc}")

        # Ensure path ends with /simple/ for pip
        path = parsed.path.rstrip("/") + "/simple/"
        final_url = urlunparse(authenticated._replace(path=path))

        logger.info("CodeArtifact authentication configured")
        return final_url

    except Exception as e:
        logger.error(f"CodeArtifact setup failed: {e}")
        raise


@app.post("/invocations")
async def invocations(request: Request):
    """
    SageMaker InvokeEndpoint sends POST requests to /invocations.
    Proxy to Ray Serve application root endpoint.

    Handles Content-Type and Accept headers as per SageMaker spec:
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html
    """
    try:
        # Get request body and headers
        body = await request.body()
        content_type = request.headers.get("Content-Type", "application/json")
        accept = request.headers.get("Accept", "application/json")

        logger.info(f"Received /invocations request: Content-Type={content_type}, Accept={accept}")

        # Prepare headers for Ray Serve
        headers = {
            "Content-Type": content_type,
            "Accept": accept,
        }

        # Proxy to Ray Serve (default route is "/")
        response = await request.app.state.client.post(
            f"{RAYSERVE_URL}/",
            content=body,
            headers=headers,
        )

        logger.info(f"Ray Serve response: status={response.status_code}")

        # Return Ray Serve response with proper headers
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("Content-Type", accept),
        )

    except httpx.TimeoutException as e:
        logger.error(f"Timeout proxying to Ray Serve: {e}")
        return Response(
            content=json.dumps({"error": "Request timeout"}).encode(),
            status_code=504,
            media_type="application/json",
        )
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ray Serve: {e}")
        return Response(
            content=json.dumps({"error": "Service unavailable"}).encode(),
            status_code=503,
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error proxying to Ray Serve: {e}")
        return Response(
            content=json.dumps({"error": str(e)}).encode(),
            status_code=500,
            media_type="application/json",
        )


@app.get("/ping")
async def ping(request: Request):
    """
    SageMaker health checks call /ping.
    Must return 200 with empty body if healthy, per SageMaker spec.

    Uses Ray Serve's built-in /-/healthz endpoint.
    """
    try:
        # Use Ray Serve's native health check endpoint
        response = await request.app.state.client.get(f"{RAYSERVE_URL}/-/healthz", timeout=5.0)

        if response.status_code == 200:
            logger.info("Health check passed")
            return PlainTextResponse(content="", status_code=200)
        else:
            logger.warning(f"Health check failed: Ray Serve returned {response.status_code}")
            return PlainTextResponse(content="", status_code=503)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return PlainTextResponse(content="", status_code=503)


if __name__ == "__main__":
    logger.info("Starting SageMaker Ray Serve Adapter on port 8080")
    logger.info(f"Proxying to Ray Serve at {RAYSERVE_URL}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )
