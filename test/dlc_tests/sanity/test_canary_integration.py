import os

import pytest

from invoke.context import Context

from test.test_utils import (
    parse_canary_images,
    get_account_id_from_image_uri,
    get_region_from_image_uri,
    is_pr_context,
    is_deep_canary_context,
    login_to_ecr_registry,
    PR_ONLY_REASON,
    PUBLIC_DLC_REGISTRY,
    LOGGER,
)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_canary_images_pullable_training(region):
    """
    Sanity test to verify canary specific functions
    """
    _run_canary_pull_test(region, "training")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_canary_images_pullable_inference(region):
    """
    Sanity test to verify canary specific functions
    """
    _run_canary_pull_test(region, "inference")


def _run_canary_pull_test(region, image_type):
    ctx = Context()
    frameworks = ("tensorflow", "mxnet", "pytorch")

    # Have a default framework to test on
    framework = "pytorch"
    for fw in frameworks:
        if fw in os.getenv("CODEBUILD_INITIATOR"):
            framework = fw
            break

    images = parse_canary_images(framework, region, image_type)
    login_to_ecr_registry(ctx, PUBLIC_DLC_REGISTRY, region)
    if not images:
        return
    for image in images.split(" "):
        ctx.run(f"docker pull {image}", hide="out")
        LOGGER.info(f"Canary image {image} is available")
        # Do not remove the pulled images as it may interfere with the functioning of the other
        # tests that need the image to be present on the build machine.


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.skipif(
    not is_deep_canary_context(), reason="This test only needs to run in deep-canary context"
)
@pytest.mark.deep_canary
@pytest.mark.model("N/A")
@pytest.mark.integration("deep_canary")
def test_deep_canary_integration(image, region):
    ctx = Context()
    image_account_id = get_account_id_from_image_uri(image)
    image_region = get_region_from_image_uri(image)
    assert image_region == region, f"Problem: Test region {region} != canary region {image_region}"

    login_to_ecr_registry(ctx, image_account_id, region)
    ctx.run(f"docker pull {image}", hide="out")
    LOGGER.info(f"Deep Canary pull test succeeded for {image}")
