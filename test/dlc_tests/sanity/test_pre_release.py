import re

import pytest

from invoke.context import Context

from test.test_utils import LOGGER


@pytest.mark.canary("Run pip check test regularly on production images")
def test_tmp_dirs(image):
    """
    Test to see if tmp dirs are empty
    """
    ctx = Context()
    container_name = f"test_tmp_dirs-{image.split(':')[-1].replace('.', '-')}"
    _start_container(container_name, image, ctx)
    _run_cmd_on_container(container_name, ctx, "ls -A /tmp")
    _run_cmd_on_container(container_name, ctx, "ls -A /var/tmp")
    _run_cmd_on_container(container_name, ctx, "ls -A ~")
    _run_cmd_on_container(container_name, ctx, "ls -A /")


def test_python_version(image):
    ctx = Context()
    container_name = f"py-version-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"
    _start_container(container_name, image, ctx)
    _run_cmd_on_container(container_name, ctx, "python --version")


def test_ubuntu_version(image):
    ctx = Context()
    container_name = f"ubuntu-version-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"
    _start_container(container_name, image, ctx)
    _run_cmd_on_container(container_name, ctx, "cat /etc/os-release")


def test_framework_version(image):
    tested_framework = None
    for framework in ("tensorflow", "mxnet", "pytorch"):
        if framework in image:
            tested_framework = framework
            break
    ctx = Context()
    container_name = f"framework-version-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"
    _start_container(container_name, image, ctx)
    _run_cmd_on_container(
        container_name, ctx, f"import {tested_framework}; {tested_framework}.__version__", executable="python"
    )


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


def _start_container(container_name, image_uri, context):
    context.run(
        f"docker run --name {container_name} -itd {image_uri}", hide=True,
    )


def _run_cmd_on_container(container_name, context, cmd, executable="bash"):
    if executable not in ("bash", "python"):
        LOGGER.warn(f"Unrecognized executable {executable}. It will be run as {executable} -c '{cmd}'")
    context.run(f"docker exec --user root {container_name} {executable} -c '{cmd}'", hide=True, timeout=30)
