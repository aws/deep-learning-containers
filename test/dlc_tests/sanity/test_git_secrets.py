import logging
import os
import sys

import pytest

from invoke.context import Context

from test.test_utils import is_pr_context, SKIP_NOT_PR_REASON

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def _recursive_find_repo_path(ctx):
    pwd = ctx.run("pwd", echo=True).stdout.strip("\n")
    repository_path = pwd
    while os.path.basename(repository_path) != "deep-learning-containers":
        repository_path = os.path.dirname(repository_path)
        if repository_path == "/":
            raise EnvironmentError(f"Repository path could not be found from {pwd}")
    return repository_path


@pytest.mark.skipif(not is_pr_context(), reason=SKIP_NOT_PR_REASON)
def test_git_secrets():
    ctx = Context()
    repository_path = os.getenv("CODEBUILD_SRC_DIR")
    if not repository_path:
        repository_path = _recursive_find_repo_path(ctx)
    LOGGER.info(f"repository_path = {repository_path}")
    with ctx.cd(repository_path):
        ctx.run("git clone https://github.com/awslabs/git-secrets.git")
        with ctx.cd("git-secrets"):
            ctx.run("make install")
        ctx.run("git secrets --install")
        ctx.run("git secrets --register-aws")
        output = ctx.run("git secrets --list", echo=True)
        LOGGER.info(f"{output.command}\n{output.stdout}")
        scan_results = ctx.run("git secrets --scan", echo=True)
        LOGGER.info(f"{scan_results.command}\n{scan_results.stdout}")
    assert scan_results.ok, scan_results.stderr
