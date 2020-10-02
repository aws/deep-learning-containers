import logging
import os
import sys

import pytest

from invoke.context import Context

from test.test_utils import PR_ONLY_REASON, get_repository_local_path, is_pr_context

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


@pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
@pytest.mark.model("N/A")
def test_git_secrets():
    ctx = Context()
    repository_path = os.getenv("CODEBUILD_SRC_DIR")
    if not repository_path:
        repository_path = get_repository_local_path()
    LOGGER.info(f"repository_path = {repository_path}")

    # Replace the regex pattern below with a matching string to run test that makes scan fail:
    SOME_FAKE_CREDENTIALS = "ASIA[A-Z0-9]{16}"
    WHITELISTED_CREDENTIALS = "AKIAIOSFODNN7EXAMPLE"
    # End of Test Section

    with ctx.cd(repository_path):
        ctx.run("git clone https://github.com/awslabs/git-secrets.git")
        with ctx.cd("git-secrets"):
            ctx.run("make install")
        ctx.run("git secrets --install")
        ctx.run("git secrets --register-aws")
        output = ctx.run("git secrets --list")
        LOGGER.info(f"\n--COMMAND--\n{output.command}\n"
                    f"--STDOUT--\n{output.stdout}\n"
                    f"--STDERR--\n{output.stderr}\n"
                    f"----------")
        scan_results = ctx.run("git secrets --scan", hide=True, warn=True)
        LOGGER.info(f"\n--COMMAND--\n{scan_results.command}\n"
                    f"--STDOUT--\n{scan_results.stdout}\n"
                    f"--STDERR--\n{scan_results.stderr}"
                    f"----------")
    assert scan_results.ok, scan_results.stderr
