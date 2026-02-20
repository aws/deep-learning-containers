import os

import pytest

from invoke.context import Context

from test.test_utils import (
    get_account_id_from_image_uri,
    get_region_from_image_uri,
    is_deep_canary_context,
    login_to_ecr_registry,
    LOGGER,
)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not is_deep_canary_context() or os.getenv("REGION") == "us-west-2",
    reason="This test only needs to run in deep-canary context",
)
@pytest.mark.deep_canary(
    "Reason: This test acts as a basic smoke check that deep-canary tests work in a region"
)
@pytest.mark.model("N/A")
@pytest.mark.integration("deep_canary")
def test_deep_canary_integration(image, region):
    ctx = Context()
    if os.getenv("IS_PUBLIC_REGISTRY_CANARY", "false").lower() == "true":
        # If the image is a public registry canary, we don't need to login to ECR
        LOGGER.info(f"Deep Canary pull test for public registry canary {image}")
        try:
            ctx.run(f"docker pull {image}", hide="out")
            LOGGER.info(f"Deep Canary pull test succeeded for {image}")
        finally:
            ctx.run(f"docker rmi {image}", warn=True, hide="out")
        return
    else:
        image_account_id = get_account_id_from_image_uri(image)
        image_region = get_region_from_image_uri(image)
        assert (
            image_region == region
        ), f"Problem: Test region {region} != canary region {image_region}"

        try:
            login_to_ecr_registry(ctx, image_account_id, region)
            ctx.run(f"docker pull {image}", hide="out")
            LOGGER.info(f"Deep Canary pull test succeeded for {image}")
        finally:
            ctx.run(f"docker rmi {image}", warn=True, hide="out")
