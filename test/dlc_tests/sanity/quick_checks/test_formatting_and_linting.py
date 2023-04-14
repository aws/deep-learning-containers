import os
import pytest

from invoke.context import Context

from test.test_utils import is_pr_context


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("python_style_formatting")
@pytest.mark.skipif(
    not is_pr_context(),
    reason="This test is only needed to validate formatting in PRs.",
)
def test_black_formatting():
    """
    Check that black style formatting exits gracefully.
    """
    # Look up the path until deep-learning-containers is in our base directory
    dlc_base_dir = os.getcwd()
    while "deep-learning-containers" not in os.path.basename(dlc_base_dir):
        dlc_base_dir = os.path.split(dlc_base_dir)[0]

    ctx = Context()

    ctx.run(f"black --check {dlc_base_dir}", hide=True)

