import logging
import os
import sys

import pytest

from invoke.context import Context

from test.test_utils import is_pr_context, PR_ONLY_REASON

pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
def test_binary_visibility(image: str):
    """
    Test to check if the binary built with image is public/private. Assumes that URIs beginning with 's3://' are private.
    This will mandate specifying all public links as ones beginning with 'https://'. While s3 objects beginning with
    'https://' may still be private, codebuild 'build' job uses 'curl' i.e. unsigned request to fetch them and hence should
    fail if an 'https://' link is still private
    :param image:
    :return:
    """
    ctx = Context()
    labels = dict(ctx.run(f"docker inspect --format='{{json .Config.Labels}}' {image}").stdout.strip())
    
    for label_name, label_value in labels.items():
        if "uri" in label_name.lower():
            assert label_value.startswith("https://")