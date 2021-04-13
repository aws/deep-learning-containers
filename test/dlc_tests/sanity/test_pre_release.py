import os
import re
import subprocess
import botocore
import boto3
from packaging.version import Version

import pytest
import requests

from urllib3.util.retry import Retry
from invoke.context import Context
from botocore.exceptions import ClientError

from src.buildspec import Buildspec
from test.test_utils import (
    LOGGER,
    CONTAINER_TESTS_PREFIX,
    ec2,
    get_container_name,
    get_framework_and_version_from_tag,
    is_canary_context,
    is_tf_version,
    is_dlc_cicd_context,
    is_pr_context,
    run_cmd_on_container,
    start_container, 
    is_time_for_canary_safety_scan, 
    is_mainline_context,
    is_nightly_context,
    get_repository_local_path,
)


@pytest.mark.model("N/A")
@pytest.mark.canary("Run stray file test regularly on production images")
def test_stray_files(image):
    """
    Test to ensure that unnecessary build artifacts are not present in any easily visible or tmp directories

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = get_container_name("test_tmp_dirs", image)
    start_container(container_name, image, ctx)

    # Running list of artifacts/artifact regular expressions we do not want in any of the directories
    stray_artifacts = [r"\.py"]

    # Running list of allowed files in the /tmp directory
    allowed_tmp_files = ["hsperfdata_root"]

    # Ensure stray artifacts are not in the tmp directory
    tmp = run_cmd_on_container(container_name, ctx, "ls -A /tmp")
    _assert_artifact_free(tmp, stray_artifacts)

    # Ensure tmp dir is empty except for whitelisted files
    tmp_files = tmp.stdout.split()
    for tmp_file in tmp_files:
        assert (
            tmp_file in allowed_tmp_files
        ), f"Found unexpected file in tmp dir: {tmp_file}. Allowed tmp files: {allowed_tmp_files}"

    # We always expect /var/tmp to be empty
    var_tmp = run_cmd_on_container(container_name, ctx, "ls -A /var/tmp")
    _assert_artifact_free(var_tmp, stray_artifacts)
    assert var_tmp.stdout.strip() == ""

    # Additional check of home and root directories to ensure that stray artifacts are not present
    home = run_cmd_on_container(container_name, ctx, "ls -A ~")
    _assert_artifact_free(home, stray_artifacts)

    root = run_cmd_on_container(container_name, ctx, "ls -A /")
    _assert_artifact_free(root, stray_artifacts)


@pytest.mark.model("N/A")
@pytest.mark.canary("Run python version test regularly on production images")
def test_python_version(image):
    """
    Check that the python version in the image tag is the same as the one on a running container.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = get_container_name("py-version", image)

    py_version = ""
    for tag_split in image.split("-"):
        if tag_split.startswith("py"):
            if len(tag_split) > 3:
                py_version = f"Python {tag_split[2]}.{tag_split[3]}"
            else:
                py_version = f"Python {tag_split[2]}"
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(container_name, ctx, "python --version")

    # Due to py2 deprecation, Python2 version gets streamed to stderr. Python installed via Conda also appears to
    # stream to stderr (in some cases).
    container_py_version = output.stdout + output.stderr

    assert py_version in container_py_version, f"Cannot find {py_version} in {container_py_version}"


@pytest.mark.model("N/A")
def test_ubuntu_version(image):
    """
    Check that the ubuntu version in the image tag is the same as the one on a running container.

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = get_container_name("ubuntu-version", image)

    ubuntu_version = ""
    for tag_split in image.split("-"):
        if tag_split.startswith("ubuntu"):
            ubuntu_version = tag_split.split("ubuntu")[-1]

    start_container(container_name, image, ctx)
    output = run_cmd_on_container(container_name, ctx, "cat /etc/os-release")
    container_ubuntu_version = output.stdout

    assert "Ubuntu" in container_ubuntu_version
    assert ubuntu_version in container_ubuntu_version


@pytest.mark.model("N/A")
@pytest.mark.canary("Run non-gpu framework version test regularly on production images")
def test_framework_version_cpu(image):
    """
    Check that the framework version in the image tag is the same as the one on a running container.
    This function tests CPU, EIA, and Neuron images.

    :param image: ECR image URI
    """
    if "gpu" in image:
        pytest.skip("GPU images will have their framework version tested in test_framework_and_cuda_version_gpu")
    if "tensorflow-inference" in image:
        pytest.skip(msg="TF inference does not have core tensorflow installed")

    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)

    # Module name is torch
    if tested_framework == "pytorch":
        tested_framework = "torch"
    ctx = Context()
    container_name = get_container_name("framework-version", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name, ctx, f"import {tested_framework}; print({tested_framework}.__version__)", executable="python"
    )
    if is_canary_context():
        assert tag_framework_version in output.stdout.strip()
    else:
        assert tag_framework_version == output.stdout.strip()


# TODO: Enable as canary once resource cleaning lambda is added
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
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
        if tested_framework == "huggingface_pytorch":
            tested_framework = "torch"
        if tested_framework == "huggingface_tensorflow":
            tested_framework = "tensorflow"
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


class DependencyCheckFailure(Exception):
    pass


def _run_dependency_check_test(image, ec2_connection, processor):
    # Record any whitelisted medium/low severity CVEs; I.E. allowed_vulnerabilities = {CVE-1000-5555, CVE-9999-9999}
    allowed_vulnerabilities = {
        # Those vulnerabilities are fixed. Current openssl version is 1.1.1g. These are false positive
        'CVE-2016-2109', 'CVE-2016-2177', 'CVE-2016-6303', 'CVE-2016-2182',
        # CVE-2020-13936: vulnerability found in apache velocity package which is a dependency for dependency-check package. Hence, ignoring.
        'CVE-2020-13936',
    }

    container_name = f"dep_check_{processor}"
    report_addon = get_container_name("depcheck-report", image)
    dependency_check_report = f"{report_addon}.html"
    html_file = f"{container_name}:/build/dependency-check-report.html"
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "testDependencyCheck")

    # Execute test, copy results to s3
    ec2.execute_ec2_training_test(ec2_connection, image, test_script, container_name=container_name)
    ec2_connection.run(f"docker cp {html_file} ~/{dependency_check_report}")
    ec2_connection.run(f"aws s3 cp ~/{dependency_check_report} s3://dlc-dependency-check")

    # Check for any vulnerabilities not mentioned in allowed_vulnerabilities
    html_output = ec2_connection.run(f"cat ~/{dependency_check_report}", hide=True).stdout
    cves = re.findall(r">(CVE-\d+-\d+)</a>", html_output)
    vulnerabilities = set(cves) - allowed_vulnerabilities

    if vulnerabilities:
        vulnerability_severity = {}

        # Check NVD for vulnerability severity to provide this useful info in error message.
        for vulnerability in vulnerabilities:
            try:
                cve_url = f"https://services.nvd.nist.gov/rest/json/cve/1.0/{vulnerability}"

                session = requests.Session()
                session.mount('https://', requests.adapters.HTTPAdapter(max_retries=Retry(total=5, status_forcelist=[404, 504, 502])))
                response = session.get(cve_url)

                if response.status_code == 200:
                    severity = (
                        response.json()
                        .get("result", {})
                        .get("CVE_Items", [{}])[0]
                        .get("impact", {})
                        .get("baseMetricV2", {})
                        .get("severity", "UNKNOWN"))
            except ConnectionError:
                LOGGER.exception(f"Failed to load NIST data for CVE {vulnerability}")

            if vulnerability_severity.get(severity):
                vulnerability_severity[severity].append(vulnerability)
            else:
                vulnerability_severity[severity] = [vulnerability]

        # TODO: Remove this once we have whitelisted appropriate LOW/MEDIUM vulnerabilities
        if not (vulnerability_severity.get("CRITICAL") or vulnerability_severity.get("HIGH")):
            return

        raise DependencyCheckFailure(
            f"Unrecognized CVEs have been reported : {vulnerability_severity}. "
            f"Allowed vulnerabilities are {allowed_vulnerabilities or None}. Please see "
            f"{dependency_check_report} for more details."
        )


@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.skipif(not (is_nightly_context() or is_mainline_context() or (is_canary_context() and is_time_for_canary_safety_scan())),
                    reason="Do not run dependency check on PR tests. "
                           "Executing test in canaries pipeline during only a limited period of time."
                    )
def test_dependency_check_cpu(cpu, ec2_connection):
    _run_dependency_check_test(cpu, ec2_connection, "cpu")


@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
@pytest.mark.skipif(not (is_nightly_context() or is_mainline_context() or (is_canary_context() and is_time_for_canary_safety_scan())),
                    reason="Do not run dependency check on PR tests. "
                           "Executing test in canaries pipeline during only a limited period of time."
                    )
def test_dependency_check_gpu(gpu, ec2_connection):
    _run_dependency_check_test(gpu, ec2_connection, "gpu")


@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skipif(not (is_nightly_context() or is_mainline_context() or (is_canary_context() and is_time_for_canary_safety_scan())),
                    reason="Do not run dependency check on PR tests. "
                           "Executing test in canaries pipeline during only a limited period of time."
                    )
def test_dependency_check_neuron(neuron, ec2_connection):
    _run_dependency_check_test(neuron, ec2_connection, "neuron")


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
    # smclarify binaries have s3fs->aiobotocore dependency which uses older version of botocore. temporarily
    # allowing this to catch other issues
    allowed_tf_exception = re.compile(
        rf"^tensorflow-serving-api{gpu_suffix} \d\.\d+\.\d+ requires "
        rf"tensorflow{gpu_suffix}, which is not installed.$"
    )
    allowed_smclarify_exception = re.compile(
        r"^aiobotocore \d+(\.\d+)* has requirement botocore<\d+(\.\d+)*,>=\d+(\.\d+)*, "
        r"but you have botocore \d+(\.\d+)*\.$"
    )

    # Add null entrypoint to ensure command exits immediately
    output = ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True, warn=True)
    if output.return_code != 0:
        if not (allowed_tf_exception.match(output.stdout) or allowed_smclarify_exception.match(output.stdout)) :
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
    container_name = get_container_name("pandas", image)
    start_container(container_name, image, ctx)

    # Make sure we can install pandas, do not fail right away if there are pip check issues
    run_cmd_on_container(container_name, ctx, "pip install pandas", warn=True)

    pandas_import_output = run_cmd_on_container(container_name, ctx, "import pandas", executable="python")

    assert (
        not pandas_import_output.stdout.strip()
    ), f"Expected no output when importing pandas, but got  {pandas_import_output.stdout}"

    # Simple import test to ensure we do not get a bz2 module import failure
    run_cmd_on_container(container_name, ctx, "import pandas; print(pandas.__version__)", executable="python")


@pytest.mark.model("N/A")
@pytest.mark.integration("emacs")
def test_emacs(image):
    """
    Ensure that emacs is installed on every image

    :param image: ECR image URI
    """
    ctx = Context()
    container_name = get_container_name("emacs", image)
    start_container(container_name, image, ctx)

    # Make sure the following emacs sanity tests exit with code 0
    run_cmd_on_container(container_name, ctx, "which emacs")
    run_cmd_on_container(container_name, ctx, "emacs -version")


@pytest.mark.model("N/A")
@pytest.mark.integration("sagemaker python sdk")
def test_sm_pysdk_2(training):
    """
    Simply verify that we have sagemaker > 2.0 in the python sdk.

    If you find that this test is failing because sm pysdk version is not greater than 2.0, then that means that
    the image under test needs to be updated.

    If you find that the training image under test does not have sagemaker pysdk, it should be added or explicitly
    skipped (with reasoning provided).

    :param training: training ECR image URI
    """

    _, image_framework_version = get_framework_and_version_from_tag(training)

    if Version(image_framework_version) == Version("1.5.0"):
        pytest.skip("sagemaker version < 2.0 is installed for PT 1.5.0 images")

    # Ensure that sm py sdk 2 is on the container
    ctx = Context()
    container_name = get_container_name("sm_pysdk", training)
    start_container(container_name, training, ctx)

    sm_version = run_cmd_on_container(
        container_name, ctx, "import sagemaker; print(sagemaker.__version__)", executable="python"
    ).stdout.strip()

    assert Version(sm_version) > Version("2"), f"Sagemaker version should be > 2.0. Found version {sm_version}"


@pytest.mark.model("N/A")
def test_cuda_paths(gpu):
    """
    Test to ensure that:
    a. buildspec contains an entry to create the same image as the image URI
    b. directory structure for GPU Dockerfiles has framework version, python version, and cuda version in it

    :param gpu: gpu image uris
    """
    image = gpu
    if "example" in image:
        pytest.skip("Skipping Example Dockerfiles which are not explicitly tied to a cuda version")

    dlc_path = os.getcwd().split("/test/")[0]
    job_type = "training" if "training" in image else "inference"

    # Ensure that image has a supported framework
    framework, framework_version = get_framework_and_version_from_tag(image)

    # Get cuda, framework version, python version through regex
    cuda_version = re.search(r"-(cu\d+)-", image).group(1)
    framework_short_version = None
    python_version = re.search(r"(py\d+)", image).group(1)
    short_python_version = None
    image_tag = re.search(
        r":(\d+(\.\d+){2}(-transformers\d+(\.\d+){2})?-(cpu|gpu|neuron)-(py\d+)(-cu\d+)-(ubuntu\d+\.\d+)(-example)?)", image
    ).group(1)

    # replacing '_' by '/' to handle huggingface_<framework> case
    framework_path = framework.replace('_', '/')
    framework_version_path = os.path.join(dlc_path, framework_path, job_type, "docker", framework_version)
    if not os.path.exists(framework_version_path):
        framework_short_version = re.match(r"(\d+.\d+)", framework_version).group(1)
        framework_version_path = os.path.join(dlc_path, framework_path, job_type, "docker", framework_short_version)
    if not os.path.exists(os.path.join(framework_version_path, python_version)):
        # Use the pyX version as opposed to the pyXY version if pyXY path does not exist
        short_python_version = python_version[:3]

    # Check buildspec for cuda version
    buildspec = "buildspec.yml"
    if is_tf_version("1", image):
        buildspec = "buildspec-tf1.yml"

    cuda_in_buildspec = False
    dockerfile_spec_abs_path = None
    cuda_in_buildspec_ref = f"CUDA_VERSION {cuda_version}"
    buildspec_path = os.path.join(dlc_path, framework_path, buildspec)
    buildspec_def = Buildspec()
    buildspec_def.load(buildspec_path)

    for name, image_spec in buildspec_def["images"].items():
        if image_spec["device_type"] == "gpu" and image_spec["tag"] == image_tag:
            cuda_in_buildspec = True
            dockerfile_spec_abs_path = os.path.join(
                os.path.dirname(framework_version_path), image_spec["docker_file"].lstrip("docker/")
            )
            break

    try:
        assert cuda_in_buildspec, f"Can't find {cuda_in_buildspec_ref} in {buildspec_path}"
    except AssertionError as e:
        if not is_dlc_cicd_context():
            LOGGER.warn(f"{e} - not failing, as this is a(n) {os.getenv('BUILD_CONTEXT', 'empty')} build context.")
        else:
            raise

    image_properties_expected_in_dockerfile_path = [
        framework_short_version or framework_version, short_python_version or python_version, cuda_version
    ]
    assert all(prop in dockerfile_spec_abs_path for prop in image_properties_expected_in_dockerfile_path), (
        f"Dockerfile location {dockerfile_spec_abs_path} does not contain all the image properties in "
        f"{image_properties_expected_in_dockerfile_path}"
    )

    assert os.path.exists(dockerfile_spec_abs_path), f"Cannot find dockerfile for {image} in {dockerfile_spec_abs_path}"


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

@pytest.mark.integration("oss_compliance")
@pytest.mark.model("N/A")
@pytest.mark.skipif(not is_dlc_cicd_context(), reason="We need to test OSS compliance only on PRs and pipelines")
def test_oss_compliance(image):
    """
    Run oss compliance check on a container to check if license attribution files exist.
    And upload source of third party packages to S3 bucket.
    """
    THIRD_PARTY_SOURCE_CODE_BUCKET = "aws-dlinfra-licenses"
    THIRD_PARTY_SOURCE_CODE_BUCKET_PATH = "third_party_source_code"
    file = "THIRD_PARTY_SOURCE_CODE_URLS"
    container_name = get_container_name("oss_compliance", image)
    context = Context()
    local_repo_path = get_repository_local_path()
    start_container(container_name, image, context)

    # run compliance test to make sure license attribution files exists. testOSSCompliance is copied as part of Dockerfile
    run_cmd_on_container(container_name, context, "/usr/local/bin/testOSSCompliance /root")

    try:
        context.run(f"docker cp {container_name}:/root/{file} {os.path.join(local_repo_path, file)}")
    finally:
        context.run(f"docker rm -f {container_name}", hide=True)

    s3_resource = boto3.resource('s3')

    with open(os.path.join(local_repo_path, file)) as source_code_file:
        for line in source_code_file:
            name, version, url = line.split(" ")
            file_name = f"{name}_v{version}_source_code"
            s3_object_path = f"{THIRD_PARTY_SOURCE_CODE_BUCKET_PATH}/{file_name}.tar.gz"
            local_file_path = os.path.join(local_repo_path, file_name)

            try:
                if not os.path.isdir(local_file_path):
                    context.run(f"git clone {url.rstrip()} {local_file_path}")
                    context.run(f"tar -czvf {local_file_path}.tar.gz {local_file_path}")
            except Exception as e:
                LOGGER.error(f"Unable to clone git repo. Error: {e}")
                raise

            try:
                if os.path.exists(f"{local_file_path}.tar.gz"):
                    LOGGER.info(f"Uploading package to s3 bucket: {line}")
                    s3_resource.Object(THIRD_PARTY_SOURCE_CODE_BUCKET, s3_object_path).load()
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    try:
                        # using aws cli as using boto3 expects to upload folder by iterating through each file instead of entire folder.
                        context.run(f"aws s3 cp {local_file_path}.tar.gz s3://{THIRD_PARTY_SOURCE_CODE_BUCKET}/{s3_object_path}")
                        object = s3_resource.Bucket(THIRD_PARTY_SOURCE_CODE_BUCKET).Object(s3_object_path)
                        object.Acl().put(ACL='public-read')
                    except ClientError as e:
                        LOGGER.error(f"Unable to upload source code to bucket {THIRD_PARTY_SOURCE_CODE_BUCKET}. Error: {e}")
                        raise
                else:
                    LOGGER.error(f"Unable to check if source code is present on bucket {THIRD_PARTY_SOURCE_CODE_BUCKET}. Error: {e}")
                    raise
