import re

import pytest

from invoke.context import Context

from test.test_utils import LOGGER, ec2, get_framework_and_version_from_tag, is_canary_context


@pytest.mark.canary("Run stray file test regularly on production images")
def test_stray_files(image):
    """
    Test to ensure that unnecessary build artifacts are not present in any easily visible or tmp directories

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = f"test_tmp_dirs-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"
    _start_container(container_name, image, ctx)

    # Running list of artifacts/artifact regular expressions we do not want in any of the directories
    stray_artifacts = [r"\.py"]

    # Running list of allowed files in the /tmp directory
    allowed_tmp_files = ["hsperfdata_root"]

    # Ensure stray artifacts are not in the tmp directory
    tmp = _run_cmd_on_container(container_name, ctx, "ls -A /tmp")
    _assert_artifact_free(tmp, stray_artifacts)

    # Ensure tmp dir is empty except for whitelisted files
    tmp_files = tmp.stdout.split()
    for tmp_file in tmp_files:
        assert tmp_file in allowed_tmp_files, f"Found unexpected file in tmp dir: {tmp_file}. " \
                                              f"Allowed tmp files: {allowed_tmp_files}"

    # We always expect /var/tmp to be empty
    var_tmp = _run_cmd_on_container(container_name, ctx, "ls -A /var/tmp")
    _assert_artifact_free(var_tmp, stray_artifacts)
    assert var_tmp.stdout.strip() == ""

    # Additional check of home and root directories to ensure that stray artifacts are not present
    home = _run_cmd_on_container(container_name, ctx, "ls -A ~")
    _assert_artifact_free(home, stray_artifacts)

    root = _run_cmd_on_container(container_name, ctx, "ls -A /")
    _assert_artifact_free(root, stray_artifacts)


@pytest.mark.canary("Run python version test regularly on production images")
def test_python_version(image):
    """
    Check that the python version in the image tag is the same as the one on a running container.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = f"py-version-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"

    py_version = ""
    for tag_split in image.split('-'):
        if tag_split.startswith('py'):
            if len(tag_split) > 3:
                py_version = f"Python {tag_split[2]}.{tag_split[3]}"
            else:
                py_version = f"Python {tag_split[2]}"
    _start_container(container_name, image, ctx)
    output = _run_cmd_on_container(container_name, ctx, "python --version")

    container_py_version = output.stdout
    # Due to py2 deprecation, Python2 version gets streamed to stderr. Python installed via Conda also appears to
    # stream to stderr, hence the pytorch condition.
    if "Python 2" in py_version or "pytorch" in image:
        container_py_version = output.stderr

    assert py_version in container_py_version, f"Cannot find {py_version} in {container_py_version}"


def test_ubuntu_version(image):
    """
    Check that the ubuntu version in the image tag is the same as the one on a running container.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = f"ubuntu-version-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"

    ubuntu_version = ""
    for tag_split in image.split('-'):
        if tag_split.startswith('ubuntu'):
            ubuntu_version = tag_split.split("ubuntu")[-1]

    _start_container(container_name, image, ctx)
    output = _run_cmd_on_container(container_name, ctx, "cat /etc/os-release")
    container_ubuntu_version = output.stdout

    assert "Ubuntu" in container_ubuntu_version
    assert ubuntu_version in container_ubuntu_version


@pytest.mark.canary("Run cpu framework version test regularly on production images")
def test_framework_version_cpu(cpu):
    """
    Check that the framework version in the image tag is the same as the one on a running container.

    :param cpu: ECR image URI with "cpu" in the name
    """
    image = cpu
    if "tensorflow" in image and "inference" in image :
        pytest.skip(msg="TF inference does not have core tensorflow installed")

    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)

    # Module name is torch
    if tested_framework == "pytorch":
        tested_framework = "torch"
    ctx = Context()
    container_name = f"framework-version-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"
    _start_container(container_name, image, ctx)
    output = _run_cmd_on_container(
        container_name, ctx, f"import {tested_framework}; print({tested_framework}.__version__)", executable="python"
    )
    if is_canary_context():
        assert tag_framework_version in output.stdout.strip()
    else:
        assert tag_framework_version == output.stdout.strip()


# TODO: Enable as canary once resource cleaning lambda is added
@pytest.mark.parametrize("ec2_instance_type", ['p2.xlarge'], indirect=True)
def test_framework_version_gpu(gpu, ec2_connection):
    """
    Check that the framework version in the image tag is the same as the one on a running container.

    :param gpu: ECR image URI with "gpu" in the name
    :param ec2_connection: fixture to establish connection with an ec2 instance
    """
    image = gpu
    if "tensorflow-inference" in image:
        pytest.skip(msg="TF inference does not have core tensorflow installed")

    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)

    # Module name is "torch"
    if tested_framework == "pytorch":
        tested_framework = "torch"
    cmd = f'import {tested_framework}; print({tested_framework}.__version__)'
    output = ec2.execute_ec2_training_test(ec2_connection, image, cmd, executable="python")

    if is_canary_context():
        assert tag_framework_version in output.stdout.strip()
    else:
        assert tag_framework_version == output.stdout.strip()


@pytest.mark.canary("Run pip check test regularly on production images")
def test_pip_check(image):
    """
    Ensure there are no broken requirements on the containers by running "pip check"

    :param image: ECR image URI
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


def test_emacs(image):
    """
    Ensure that emacs is installed on every image

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = f"emacs-{image.split('/')[-1].replace('.', '-').replace(':', '-')}"
    _start_container(container_name, image, ctx)

    # Make sure the following emacs sanity tests exit with code 0
    _run_cmd_on_container(container_name, ctx, "which emacs")
    _run_cmd_on_container(container_name, ctx, "emacs -version")


def _start_container(container_name, image_uri, context):
    """
    Helper function to start a container locally

    :param container_name: Name of the docker container
    :param image_uri: ECR image URI
    :param context: Invoke context object
    """
    context.run(
        f"docker run --entrypoint='/bin/bash' --name {container_name} -itd {image_uri}", hide=True,
    )


def _run_cmd_on_container(container_name, context, cmd, executable="bash"):
    """
    Helper function to run commands on a locally running container

    :param container_name: Name of the docker container
    :param context: ECR image URI
    :param cmd: Command to run on the container
    :param executable: Executable to run on the container (bash or python)
    :return: invoke output, can be used to parse stdout, etc
    """
    if executable not in ("bash", "python"):
        LOGGER.warn(f"Unrecognized executable {executable}. It will be run as {executable} -c '{cmd}'")
    return context.run(f"docker exec --user root {container_name} {executable} -c '{cmd}'", hide=True, timeout=30)


def _assert_artifact_free(output, stray_artifacts):
    """
    Manage looping through assertions to determine that directories don't have known stray files.

    :param output: Invoke result object
    :param stray_artifacts: List of things that should not be present in these directories
    """
    for artifact in stray_artifacts:
        assert not re.search(artifact, output.stdout), \
            f"Matched {artifact} in {output.stdout} while running {output.command}"
