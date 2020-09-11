import os
import re

import pytest

from invoke.context import Context

from test.test_utils import (
    LOGGER,
    CONTAINER_TESTS_PREFIX,
    ec2,
    get_framework_and_version_from_tag,
    is_canary_context,
    is_tf1,
    is_dlc_cicd_context
)


@pytest.mark.model("N/A")
@pytest.mark.canary("Run stray file test regularly on production images")
def test_stray_files(image):
    """
    Test to ensure that unnecessary build artifacts are not present in any easily visible or tmp directories

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = _get_container_name("test_tmp_dirs", image)
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
        assert (
            tmp_file in allowed_tmp_files
        ), f"Found unexpected file in tmp dir: {tmp_file}. Allowed tmp files: {allowed_tmp_files}"

    # We always expect /var/tmp to be empty
    var_tmp = _run_cmd_on_container(container_name, ctx, "ls -A /var/tmp")
    _assert_artifact_free(var_tmp, stray_artifacts)
    assert var_tmp.stdout.strip() == ""

    # Additional check of home and root directories to ensure that stray artifacts are not present
    home = _run_cmd_on_container(container_name, ctx, "ls -A ~")
    _assert_artifact_free(home, stray_artifacts)

    root = _run_cmd_on_container(container_name, ctx, "ls -A /")
    _assert_artifact_free(root, stray_artifacts)


@pytest.mark.model("N/A")
@pytest.mark.canary("Run python version test regularly on production images")
def test_python_version(image):
    """
    Check that the python version in the image tag is the same as the one on a running container.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = _get_container_name("py-version", image)

    py_version = ""
    for tag_split in image.split("-"):
        if tag_split.startswith("py"):
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


@pytest.mark.model("N/A")
def test_ubuntu_version(image):
    """
    Check that the ubuntu version in the image tag is the same as the one on a running container.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = _get_container_name("ubuntu-version", image)

    ubuntu_version = ""
    for tag_split in image.split("-"):
        if tag_split.startswith("ubuntu"):
            ubuntu_version = tag_split.split("ubuntu")[-1]

    _start_container(container_name, image, ctx)
    output = _run_cmd_on_container(container_name, ctx, "cat /etc/os-release")
    container_ubuntu_version = output.stdout

    assert "Ubuntu" in container_ubuntu_version
    assert ubuntu_version in container_ubuntu_version


@pytest.mark.model("N/A")
@pytest.mark.canary("Run cpu framework version test regularly on production images")
def test_framework_version_cpu(cpu):
    """
    Check that the framework version in the image tag is the same as the one on a running container.

    :param cpu: ECR image URI with "cpu" in the name
    """
    image = cpu
    if "tensorflow-inference" in image:
        pytest.skip(msg="TF inference does not have core tensorflow installed")

    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)

    # Module name is torch
    if tested_framework == "pytorch":
        tested_framework = "torch"
    ctx = Context()
    container_name = _get_container_name("framework-version", image)
    _start_container(container_name, image, ctx)
    output = _run_cmd_on_container(
        container_name, ctx, f"import {tested_framework}; print({tested_framework}.__version__)", executable="python"
    )
    if is_canary_context():
        assert tag_framework_version in output.stdout.strip()
    else:
        assert tag_framework_version == output.stdout.strip()


# TODO: Enable as canary once resource cleaning lambda is added
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["p2.xlarge"], indirect=True)
def test_framework_and_cuda_version_gpu(gpu, ec2_connection):
    """
    Check that the framework  and cuda version in the image tag is the same as the one on a running container.

    :param gpu: ECR image URI with "gpu" in the name
    :param ec2_connection: fixture to establish connection with an ec2 instance
    """
    image = gpu
    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)

    # Framework Version Check #
    # Skip framework version test for tensorflow-inference, since it doesn't have core TF installed
    if "tensorflow-inference" not in image:
        # Module name is "torch"
        if tested_framework == "pytorch":
            tested_framework = "torch"
        cmd = f"import {tested_framework}; print({tested_framework}.__version__)"
        output = ec2.execute_ec2_training_test(ec2_connection, image, cmd, executable="python")

        if is_canary_context():
            assert tag_framework_version in output.stdout.strip()
        else:
            assert tag_framework_version == output.stdout.strip()

    # CUDA Version Check #
    cuda_version = re.search(r"-cu(\d+)-", image).group(1)

    # MXNet inference containers do not currently have nvcc in /usr/local/cuda/bin, so check symlink
    if "mxnet-inference" in image:
        cuda_cmd = "readlink /usr/local/cuda"
    else:
        cuda_cmd = "nvcc --version"
    cuda_output = ec2.execute_ec2_training_test(ec2_connection, image, cuda_cmd, container_name="cuda_version_test")

    # Ensure that cuda version in tag is in the container
    assert cuda_version in cuda_output.stdout.replace(".", "")


@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_dependency_check_cpu(cpu, ec2_connection):
    container_name = "dep_check_cpu"
    report_addon = _get_container_name('depcheck-report', cpu)
    dependency_check_report = f"{report_addon}.html"
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, 'testDependencyCheck')
    ec2.execute_ec2_training_test(ec2_connection, cpu, test_script, container_name=container_name)

    if is_dlc_cicd_context():
        ec2_connection.run(f"docker cp {container_name}:/build/dependency-check-report.html ~/{dependency_check_report}")
        ec2_connection.run(f"aws s3 cp ~/{dependency_check_report} s3://dlc-dependency-check")


@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_dependency_check_gpu(gpu, ec2_connection):
    container_name = "dep_check_gpu"
    report_addon = _get_container_name('depcheck-report', gpu)
    dependency_check_report = f"{report_addon}.html"
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, 'testDependencyCheck')
    ec2.execute_ec2_training_test(ec2_connection, gpu, test_script, container_name=container_name)

    if is_dlc_cicd_context():
        ec2_connection.run(f"docker cp {container_name}:/build/dependency-check-report.html ~/{dependency_check_report}")
        ec2_connection.run(f"aws s3 cp ~/{dependency_check_report} s3://dlc-dependency-check")


@pytest.mark.model("N/A")
@pytest.mark.canary("Run pip check test regularly on production images")
def test_pip_check(image):
    """
    Ensure there are no broken requirements on the containers by running "pip check"

    :param image: ECR image URI
    """
    ctx = Context()
    gpu_suffix = "-gpu" if "gpu" in image else ""

    # TF inference containers do not have core tensorflow installed by design. Allowing for this pip check error
    # to occur in order to catch other pip check issues that may be associated with TF inference
    allowed_exception = re.compile(
        rf"^tensorflow-serving-api{gpu_suffix} \d\.\d+\.\d+ requires "
        rf"tensorflow{gpu_suffix}, which is not installed.$"
    )

    # Add null entrypoint to ensure command exits immediately
    output = ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True, warn=True)
    if output.return_code != 0:
        if not allowed_exception.match(output.stdout):
            # Rerun pip check test if this is an unexpected failure
            ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True)


@pytest.mark.model("N/A")
@pytest.mark.integration("pandas")
def test_pandas(image):
    """
    It's possible that in newer python versions, we may have issues with installing pandas due to lack of presence
    of the bz2 module in py3 containers. This is a sanity test to ensure that pandas import works properly in all
    containers.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = _get_container_name("pandas", image)
    _start_container(container_name, image, ctx)

    # Make sure we can install pandas, do not fail right away if there are pip check issues
    _run_cmd_on_container(container_name, ctx, "pip install pandas", warn=True)

    pandas_import_output = _run_cmd_on_container(container_name, ctx, "import pandas", executable="python")

    assert (
        not pandas_import_output.stdout.strip()
    ), f"Expected no output when importing pandas, but got  {pandas_import_output.stdout}"

    # Simple import test to ensure we do not get a bz2 module import failure
    _run_cmd_on_container(container_name, ctx, "import pandas; print(pandas.__version__)", executable="python")


@pytest.mark.model("N/A")
@pytest.mark.integration("emacs")
def test_emacs(image):
    """
    Ensure that emacs is installed on every image

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = _get_container_name("emacs", image)
    _start_container(container_name, image, ctx)

    # Make sure the following emacs sanity tests exit with code 0
    _run_cmd_on_container(container_name, ctx, "which emacs")
    _run_cmd_on_container(container_name, ctx, "emacs -version")


@pytest.mark.model("N/A")
def test_cuda_paths(gpu):
    """
    Test to ensure directory structure for GPU Dockerfiles has cuda version in it

    :param gpu: gpu image uris
    """
    image = gpu
    if "example" in image:
        pytest.skip("Skipping Example Dockerfiles which are not explicitly tied to a cuda version")

    dlc_path = os.getcwd().split("/test/")[0]
    job_type = "training" if "training" in image else "inference"

    # Ensure that image has a supported framework
    frameworks = ("tensorflow", "pytorch", "mxnet")
    framework = ""
    for fw in frameworks:
        if fw in image:
            framework = fw
            break
    assert framework, f"Cannot find any frameworks {frameworks} in image uri {image}"

    # Get cuda, framework version, python version through regex
    cuda_version = re.search(r"-(cu\d+)-", image).group(1)
    framework_version = re.search(r":(\d+(.\d+){2})", image).group(1)
    python_version = re.search(r"(py\d+)", image).group(1)

    framework_version_path = os.path.join(dlc_path, framework, job_type, "docker", framework_version)
    if not os.path.exists(os.path.join(framework_version_path, python_version)):
        # Use the pyX version as opposed to the pyXY version if pyXY path does not exist
        python_version = python_version[:3]

    # Check buildspec for cuda version
    buildspec = "buildspec.yml"
    if is_tf1(image):
        buildspec = "buildspec-tf1.yml"

    cuda_in_buildspec = False
    cuda_in_buildspec_ref = f"CUDA_VERSION {cuda_version}"
    buildspec_path = os.path.join(dlc_path, framework, buildspec)
    with open(buildspec_path, "r") as bf:
        for line in bf:
            if cuda_in_buildspec_ref in line:
                cuda_in_buildspec = True
                break

    try:
        assert cuda_in_buildspec, f"Can't find {cuda_in_buildspec_ref} in {buildspec_path}"
    except AssertionError as e:
        if not is_dlc_cicd_context():
            LOGGER.warn(f"{e} - not failing, as this is a(n) {os.getenv('BUILD_CONTEXT', 'empty')} build context.")
        else:
            raise

    # Check that a Dockerfile exists in the right directory
    dockerfile_path = os.path.join(framework_version_path, python_version, cuda_version, "Dockerfile.gpu")

    assert os.path.exists(dockerfile_path), f"Cannot find dockerfile for image {image} in {dockerfile_path}"


def _get_container_name(prefix, image_uri):
    """
    Create a unique container name based off of a test related prefix and the image uri

    :param prefix: test related prefix, like "emacs" or "pip-check"
    :param image_uri: ECR image URI
    :return: container name
    """
    return f"{prefix}-{image_uri.split('/')[-1].replace('.', '-').replace(':', '-')}"


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


def _run_cmd_on_container(container_name, context, cmd, executable="bash", warn=False):
    """
    Helper function to run commands on a locally running container

    :param container_name: Name of the docker container
    :param context: ECR image URI
    :param cmd: Command to run on the container
    :param executable: Executable to run on the container (bash or python)
    :param warn: Whether to only warn as opposed to exit if command fails
    :return: invoke output, can be used to parse stdout, etc
    """
    if executable not in ("bash", "python"):
        LOGGER.warn(f"Unrecognized executable {executable}. It will be run as {executable} -c '{cmd}'")
    return context.run(
        f"docker exec --user root {container_name} {executable} -c '{cmd}'", hide=True, warn=warn, timeout=60
    )


def _assert_artifact_free(output, stray_artifacts):
    """
    Manage looping through assertions to determine that directories don't have known stray files.

    :param output: Invoke result object
    :param stray_artifacts: List of things that should not be present in these directories
    """
    for artifact in stray_artifacts:
        assert not re.search(
            artifact, output.stdout
        ), f"Matched {artifact} in {output.stdout} while running {output.command}"
