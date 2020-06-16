import re

import pytest

from invoke.context import Context


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
    output = ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True, warn=True)
    if output.return_code != 0:
        if not allowed_exception.match(output.stdout):
            # Rerun pip check test if this is an unexpected failure
            ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True)


