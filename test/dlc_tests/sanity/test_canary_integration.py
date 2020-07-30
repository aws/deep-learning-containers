import os

import pytest

from invoke.context import Context

from test.test_utils import (
    parse_canary_images,
    is_pr_context,
    login_to_ecr_registry,
    PR_ONLY_REASON,
    PUBLIC_DLC_REGISTRY,
)


@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_canary_images_pullable(region):
    """
    Sanity test to verify canary specific functions
    """
    ctx = Context()
    frameworks = ("tensorflow", "mxnet", "pytorch")

    # Have a default framework to test on
    framework = "pytorch"
    for fw in frameworks:
        if fw in os.getenv("CODEBUILD_INITIATOR"):
            framework = fw
            break

    images = parse_canary_images(framework, region)
    login_to_ecr_registry(ctx, PUBLIC_DLC_REGISTRY, region)
    for image in images.split(" "):
        ctx.run(f"docker pull {image}")
