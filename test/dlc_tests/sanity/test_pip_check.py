import re

import pytest

from invoke.context import Context
from invoke.exceptions import UnexpectedExit


@pytest.mark.canary("Run pip check test regularly on production images")
def test_pip_check(image):
    """
    Test to run pip sanity tests
    """
    # Add null entrypoint to ensure command exits immediately
    ctx = Context()
    gpu_suffix = '-gpu' if 'gpu' in image else ''

    # TF inference containers do not have core tensorflow installed by design. Allowing for this pip check error
    # to occur in order to catch other pip check issues that may be associated with TF inference
    allowed_exception = re.compile(rf'^tensorflow-serving-api{gpu_suffix} \d\.\d+\.\d+ requires '
                                   rf'tensorflow{gpu_suffix}, which is not installed.$')
    try:
        run_out = ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True)
    except UnexpectedExit as e:
        if not allowed_exception.match(run_out):
            raise

