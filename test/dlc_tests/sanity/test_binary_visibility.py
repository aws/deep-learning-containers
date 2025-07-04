import json
from test.test_utils import (
    PR_ONLY_REASON,
    get_framework_and_version_from_tag,
    is_pr_context,
    is_trcomp_image,
)

import pytest
from invoke.context import Context
from packaging.specifiers import SpecifierSet
from packaging.version import Version


@pytest.mark.usefixtures("sagemaker", "functionality_sanity")
@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_binary_visibility(image: str):
    """
    Test to check if the binary built with image is public/private. Assumes that URIs beginning with 's3://' are private.
    This will mandate specifying all public links as ones beginning with 'https://'. While s3 objects beginning with
    'https://' may still be private, codebuild 'build' job uses 'curl' i.e. unsigned request to fetch them and hence should
    fail if an 'https://' link is still private
    """

    framework, version = get_framework_and_version_from_tag(image)
    if (
        is_trcomp_image(image)
        and framework == "huggingface_tensorflow_trcomp"
        and Version(version) in SpecifierSet("==2.6.*")
    ):
        pytest.skip("Skipping test for HF TrComp Tensorflow 2.6 images")

    ctx = Context()
    labels = json.loads(
        ctx.run("docker inspect --format='{{json .Config.Labels}}' " + image).stdout.strip()
    )

    for label_name, label_value in labels.items():
        if "uri" in label_name.lower():
            assert label_value.startswith("https://")
