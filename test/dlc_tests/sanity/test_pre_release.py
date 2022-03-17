import os
import re
import subprocess
import botocore
import boto3
import time

from packaging.version import Version
from packaging.specifiers import SpecifierSet

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
    get_neuron_framework_and_version_from_tag,
    is_canary_context,
    is_tf_version,
    is_dlc_cicd_context,
    is_pr_context,
    run_cmd_on_container,
    start_container,
    stop_and_remove_container,
    is_time_for_canary_safety_scan,
    is_mainline_context,
    is_nightly_context,
    get_repository_local_path,
    get_repository_and_tag_from_image_uri,
    get_python_version_from_image_uri,
    is_tf_version,
    get_processor_from_image_uri,
    UL18_CPU_ARM64_US_WEST_2,
    UBUNTU_18_HPU_DLAMI_US_WEST_2
)


@pytest.mark.usefixtures("sagemaker")
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


@pytest.mark.usefixtures("sagemaker")
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


@pytest.mark.usefixtures("sagemaker")
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


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run non-gpu framework version test regularly on production images")
def test_framework_version_cpu(image):
    """
    Check that the framework version in the image tag is the same as the one on a running container.
    This function tests CPU, EIA images.

    :param image: ECR image URI
    """
    if "gpu" in image:
        pytest.skip(
            "GPU images will have their framework version tested in test_framework_and_cuda_version_gpu")
    if "neuron" in image:
        pytest.skip(
            "Neuron images will have their framework version tested in test_framework_and_neuron_sdk_version")
    image_repo_name, _ = get_repository_and_tag_from_image_uri(image)
    if re.fullmatch(r"(pr-|beta-|nightly-)?tensorflow-inference(-eia|-graviton)?", image_repo_name):
        pytest.skip(
            msg="TF inference for CPU/GPU/EIA does not have core tensorflow installed")

    tested_framework, tag_framework_version = get_framework_and_version_from_tag(
        image)

    # Framework name may include huggingface
    if tested_framework.startswith('huggingface_'):
        tested_framework = tested_framework[len("huggingface_"):]
    # Module name is torch
    if tested_framework == "pytorch":
        tested_framework = "torch"
    elif tested_framework == "autogluon":
        tested_framework = "autogluon.core"
    ctx = Context()
    container_name = get_container_name("framework-version", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name, ctx, f"import {tested_framework}; print({tested_framework}.__version__)", executable="python"
    )
    if is_canary_context():
        assert tag_framework_version in output.stdout.strip()
    else:
        if tested_framework == "autogluon.core":
            version_to_check = "0.3.1" if tag_framework_version == "0.3.2" else tag_framework_version
            assert output.stdout.strip().startswith(version_to_check)
        # Habana v1.2 binary does not follow the X.Y.Z+cpu naming convention
        elif "habana" not in image_repo_name:
            if tested_framework == "torch" and Version(tag_framework_version) >= Version("1.10.0"):
                torch_version_pattern = r"{torch_version}(\+cpu)".format(torch_version=tag_framework_version)
                assert re.fullmatch(torch_version_pattern, output.stdout.strip()), (
                    f"torch.__version__ = {output.stdout.strip()} does not match {torch_version_pattern}\n"
                    f"Please specify framework version as X.Y.Z+cpu"
                )
        else:
            if "neuron" in image:
                assert tag_framework_version in output.stdout.strip()
            if all(_string in image for _string in ["pytorch", "habana", "synapseai1.3.0"]):
                # Habana Pytorch version looks like 1.10.0a0+gitb488e78 for SynapseAI1.3
                pt_fw_version_pattern = r"(\d+(\.\d+){1,2}(-rc\d)?)((a0\+git\w{7}))"
                pt_fw_version_match = re.fullmatch(pt_fw_version_pattern, output.stdout.strip())
                assert tag_framework_version == pt_fw_version_match.group(1)
            else:
                assert tag_framework_version == output.stdout.strip()
    stop_and_remove_container(container_name, ctx)


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
def test_framework_and_neuron_sdk_version(neuron):
    """
    Gets the neuron sdk tag from the image. For that neuron sdk and the frame work version from
    the image, it gets the expected frame work version. Then checks that the expected framework version
    same as the one on a running container.
    This function test only Neuron images.

    :param image: ECR image URI
    """
    image = neuron

    tested_framework, neuron_tag_framework_version = get_neuron_framework_and_version_from_tag(image)

    # neuron tag is there in pytorch images for now. Once all frameworks have it, then this will
    # be removed
    if neuron_tag_framework_version is None:
        if tested_framework is "pytorch":
            assert neuron_tag_framework_version != None
        else:
            pytest.skip(msg="Neuron SDK tag is not there as part of image")

    # Framework name may include huggingface
    if tested_framework.startswith('huggingface_'):
        tested_framework = tested_framework[len("huggingface_"):]

    if tested_framework == "pytorch":
        tested_framework = "torch_neuron"
    elif tested_framework == "tensorflow":
        tested_framework = "tensorflow_neuron"
    elif tested_framework == "mxnet":
        tested_framework = "mxnet"

    ctx = Context()

    container_name = get_container_name("framework-version-neuron", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name, ctx, f"import {tested_framework}; print({tested_framework}.__version__)", executable="python"
    )

    if tested_framework == "mxnet":
        # TODO -For neuron the mx_neuron module does not support the __version__ yet and we
        # can get the version of only the base mxnet model. The base mxnet model just
        # has framework version and does not have the neuron semantic version yet. Till
        # the mx_neuron supports __version__ do the minimal check and not exact match
        _ , tag_framework_version = get_framework_and_version_from_tag(image)
        assert tag_framework_version == output.stdout.strip()
    else:
        assert neuron_tag_framework_version == output.stdout.strip()
    stop_and_remove_container(container_name, ctx)



# TODO: Enable as canary once resource cleaning lambda is added
@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_framework_and_cuda_version_gpu(gpu, ec2_connection):
    """
    Check that the framework  and cuda version in the image tag is the same as the one on a running container.

    :param gpu: ECR image URI with "gpu" in the name
    :param ec2_connection: fixture to establish connection with an ec2 instance
    """
    image = gpu
    tested_framework, tag_framework_version = get_framework_and_version_from_tag(
        image)

    # Framework Version Check #
    # Skip framework version test for tensorflow-inference, since it doesn't have core TF installed
    if "tensorflow-inference" not in image:
        # Framework name may include huggingface
        if tested_framework.startswith('huggingface_'):
            tested_framework = tested_framework[len("huggingface_"):]
        # Module name is "torch"
        if tested_framework == "pytorch":
            tested_framework = "torch"
        elif tested_framework == "autogluon":
            tested_framework = "autogluon.core"
        cmd = f"import {tested_framework}; print({tested_framework}.__version__)"
        output = ec2.execute_ec2_training_test(ec2_connection, image, cmd, executable="python")

        if is_canary_context():
            assert tag_framework_version in output.stdout.strip()
        else:
            if tested_framework == "autogluon.core":
                version_to_check = "0.3.1" if tag_framework_version == "0.3.2" else tag_framework_version
                assert output.stdout.strip().startswith(version_to_check)
            elif tested_framework == "torch" and Version(tag_framework_version) >= Version("1.10.0"):
                torch_version_pattern = r"{torch_version}(\+cu\d+)".format(torch_version=tag_framework_version)
                assert re.fullmatch(torch_version_pattern, output.stdout.strip()), (
                    f"torch.__version__ = {output.stdout.strip()} does not match {torch_version_pattern}\n"
                    f"Please specify framework version as X.Y.Z+cuXXX"
                )
            else:
                assert tag_framework_version == output.stdout.strip()

    # CUDA Version Check #
    cuda_version = re.search(r"-cu(\d+)-", image).group(1)

    # MXNet inference/HF tensorflow inference and Autogluon containers do not currently have nvcc in /usr/local/cuda/bin, so check symlink
    if "mxnet-inference" in image or "autogluon" in image or "huggingface-tensorflow-inference" in image:
        cuda_cmd = "readlink /usr/local/cuda"
    else:
        cuda_cmd = "nvcc --version"
    cuda_output = ec2.execute_ec2_training_test(
        ec2_connection, image, cuda_cmd, container_name="cuda_version_test")

    # Ensure that cuda version in tag is in the container
    assert cuda_version in cuda_output.stdout.replace(".", "")


class DependencyCheckFailure(Exception):
    pass


def _run_dependency_check_test(image, ec2_connection):
    # Record any whitelisted medium/low severity CVEs; I.E. allowed_vulnerabilities = {CVE-1000-5555, CVE-9999-9999}
    allowed_vulnerabilities = {
        # Those vulnerabilities are fixed. Current openssl version is 1.1.1g. These are false positive
        "CVE-2016-2109",
        "CVE-2016-2177",
        "CVE-2016-6303",
        "CVE-2016-2182",
    }

    processor = get_processor_from_image_uri(image)

    # Whitelist CVE #CVE-2021-3711 for DLCs where openssl is installed using apt-get
    framework, _ = get_framework_and_version_from_tag(image)
    short_fw_version = re.search(r"(\d+\.\d+)", image).group(1)

    # Check that these versions have been matched on https://ubuntu.com/security/CVE-2021-3711 before adding
    allow_openssl_cve_fw_versions = {
        "tensorflow": {
            "1.15": ["cpu", "gpu", "neuron"],
            "2.3": ["cpu", "gpu"],
            "2.4": ["cpu", "gpu"],
            "2.5": ["cpu", "gpu", "neuron"],
            "2.6": ["cpu", "gpu"],
            "2.7": ["cpu", "gpu", "hpu"],
            "2.8": ["cpu", "gpu"],
        },
        "mxnet": {"1.8": ["neuron"], "1.9": ["cpu", "gpu"]},
        "pytorch": {"1.8": ["cpu", "gpu"], "1.10": ["cpu", "hpu"]},
        "huggingface_pytorch": {"1.8": ["cpu", "gpu"], "1.9": ["cpu", "gpu"]},
        "huggingface_tensorflow": {"2.4": ["cpu", "gpu"], "2.5": ["cpu", "gpu"]},
        "autogluon": {"0.3": ["cpu"]},
    }

    if processor in allow_openssl_cve_fw_versions.get(framework, {}).get(short_fw_version, []):
        allowed_vulnerabilities.add("CVE-2021-3711")

    container_name = f"dep_check_{processor}"
    report_addon = get_container_name("depcheck-report", image)
    dependency_check_report = f"{report_addon}.html"
    html_file = f"{container_name}:/build/dependency-check-report.html"
    test_script = os.path.join(CONTAINER_TESTS_PREFIX, "testDependencyCheck")

    # Execute test, copy results to s3
    ec2.execute_ec2_training_test(
        ec2_connection, image, test_script, container_name=container_name, bin_bash_entrypoint=True
    )
    ec2_connection.run(f"docker cp {html_file} ~/{dependency_check_report}")
    ec2_connection.run(
        f"aws s3 cp ~/{dependency_check_report} s3://dlc-dependency-check")

    # Check for any vulnerabilities not mentioned in allowed_vulnerabilities
    html_output = ec2_connection.run(
        f"cat ~/{dependency_check_report}", hide=True).stdout
    cves = re.findall(r">(CVE-\d+-\d+)</a>", html_output)
    vulnerabilities = set(cves) - allowed_vulnerabilities

    if vulnerabilities:
        vulnerability_severity = {}

        # Check NVD for vulnerability severity to provide this useful info in error message.
        for vulnerability in vulnerabilities:
            try:
                cve_url = f"https://services.nvd.nist.gov/rest/json/cve/1.0/{vulnerability}"

                session = requests.Session()
                session.mount(
                    "https://",
                    requests.adapters.HTTPAdapter(max_retries=Retry(
                        total=5, status_forcelist=[404, 504, 502])),
                )
                response = session.get(cve_url)

                if response.status_code == 200:
                    severity = (
                        response.json()
                        .get("result", {})
                        .get("CVE_Items", [{}])[0]
                        .get("impact", {})
                        .get("baseMetricV2", {})
                        .get("severity", "UNKNOWN")
                    )
                    if vulnerability_severity.get(severity):
                        vulnerability_severity[severity].append(vulnerability)
                    else:
                        vulnerability_severity[severity] = [vulnerability]
            except ConnectionError:
                LOGGER.exception(
                    f"Failed to load NIST data for CVE {vulnerability}")

        # TODO: Remove this once we have whitelisted appropriate LOW/MEDIUM vulnerabilities
        if not (vulnerability_severity.get("CRITICAL") or vulnerability_severity.get("HIGH")):
            return

        raise DependencyCheckFailure(
            f"Unrecognized CVEs have been reported : {vulnerability_severity}. "
            f"Allowed vulnerabilities are {allowed_vulnerabilities or None}. Please see "
            f"{dependency_check_report} for more details."
        )


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.skipif(
    (is_canary_context() and not is_time_for_canary_safety_scan()),
    reason="Executing test in canaries pipeline during only a limited period of time.",
)
def test_dependency_check_cpu(cpu, ec2_connection, cpu_only, x86_compatible_only):
    _run_dependency_check_test(cpu, ec2_connection)


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
@pytest.mark.skipif(
    (is_canary_context() and not is_time_for_canary_safety_scan()),
    reason="Executing test in canaries pipeline during only a limited period of time.",
)
def test_dependency_check_gpu(gpu, ec2_connection, gpu_only):
    _run_dependency_check_test(gpu, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.skipif(
    (is_canary_context() and not is_time_for_canary_safety_scan()),
    reason="Executing test in canaries pipeline during only a limited period of time.",
)
def test_dependency_check_eia(eia, ec2_connection):
    _run_dependency_check_test(eia, ec2_connection)

@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["dl1.24xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True)
def test_dependency_check_hpu(hpu, ec2_connection):
    _run_dependency_check_test(hpu, ec2_connection)


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skipif(
    (is_canary_context() and not is_time_for_canary_safety_scan()),
    reason="Executing test in canaries pipeline during only a limited period of time.",
)
def test_dependency_check_neuron(neuron, ec2_connection):
    _run_dependency_check_test(neuron, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run dependency tests regularly on production images")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UL18_CPU_ARM64_US_WEST_2], indirect=True)
@pytest.mark.skipif(
    (is_canary_context() and not is_time_for_canary_safety_scan()),
    reason="Executing test in canaries pipeline during only a limited period of time.",
)
def test_dependency_check_graviton_cpu(cpu, ec2_connection, graviton_compatible_only):
    _run_dependency_check_test(cpu, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
def test_dataclasses_check(image):
    """
    Ensure there is no dataclasses pip package is installed for python 3.7 and above version.
    Python version retrieved from the ecr image uri is expected in the format `py<major_verion><minor_version>`
    :param image: ECR image URI
    """
    ctx = Context()
    pip_package = "dataclasses"

    container_name = get_container_name("dataclasses-check", image)

    python_version = get_python_version_from_image_uri(image).replace("py","")
    python_version = int(python_version)

    if python_version >= 37:
        start_container(container_name, image, ctx)
        output = run_cmd_on_container(
            container_name, ctx, f"pip show {pip_package}", warn=True)

        if output.return_code == 0:
            pytest.fail(
                f"{pip_package} package exists in the DLC image {image} that has py{python_version} version which is greater than py36 version")
        else:
            LOGGER.info(
                f"{pip_package} package does not exists in the DLC image {image}")
    else:
        pytest.skip(f"Skipping test for DLC image {image} that has py36 version as {pip_package} is not included in the python framework")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run pip check test regularly on production images")
def test_pip_check(image):
    """
    Ensure there are no broken requirements on the containers by running "pip check"

    :param image: ECR image URI
    """
    ctx = Context()
    gpu_suffix = "-gpu" if "gpu" in image else ""
    allowed_exception_list = []

    # TF inference containers do not have core tensorflow installed by design. Allowing for this pip check error
    # to occur in order to catch other pip check issues that may be associated with TF inference
    # smclarify binaries have s3fs->aiobotocore dependency which uses older version of botocore. temporarily
    # allowing this to catch other issues
    allowed_tf_exception = re.compile(
        rf"^tensorflow-serving-api{gpu_suffix} \d\.\d+\.\d+ requires tensorflow{gpu_suffix}, which is not installed.$"
    )
    allowed_exception_list.append(allowed_tf_exception)

    allowed_smclarify_exception = re.compile(
        r"^aiobotocore \d+(\.\d+)* has requirement botocore<\d+(\.\d+)*,>=\d+(\.\d+)*, "
        r"but you have botocore \d+(\.\d+)*\.$"
    )
    allowed_exception_list.append(allowed_smclarify_exception)

    # The v0.22 version of tensorflow-io has a bug fixed in v0.23 https://github.com/tensorflow/io/releases/tag/v0.23.0
    allowed_habana_tf_exception = re.compile(rf"^tensorflow-io 0.22.0 requires tensorflow, which is not installed.$")
    allowed_exception_list.append(allowed_habana_tf_exception)

    framework, framework_version = get_framework_and_version_from_tag(image)
    # The v0.21 version of tensorflow-io has a bug fixed in v0.23 https://github.com/tensorflow/io/releases/tag/v0.23.0
    if framework == "tensorflow" and Version(framework_version) in SpecifierSet(">=2.6.3,<2.7"):
        allowed_tf263_exception = re.compile(rf"^tensorflow-io 0.21.0 requires tensorflow, which is not installed.$")
        allowed_exception_list.append(allowed_tf263_exception)

    if "autogluon" in image and (("0.3.1" in image) or ("0.3.2" in image)):
        allowed_autogluon_exception = re.compile(
            rf"autogluon-(vision|mxnet) 0.3.1 has requirement Pillow<8.4.0,>=8.3.0, but you have pillow \d+(\.\d+)*"
        )
        allowed_exception_list.append(allowed_autogluon_exception)

    # Add null entrypoint to ensure command exits immediately
    output = ctx.run(
        f"docker run --entrypoint='' {image} pip check", hide=True, warn=True)
    if output.return_code != 0:
        if not(any([allowed_exception.match(output.stdout) for allowed_exception in allowed_exception_list])):
            # Rerun pip check test if this is an unexpected failure
            ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True)


@pytest.mark.usefixtures("sagemaker", "huggingface")
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
        pytest.skip(
            "Skipping Example Dockerfiles which are not explicitly tied to a cuda version")

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
        r":(\d+(\.\d+){2}(-transformers\d+(\.\d+){2})?-(gpu)-(py\d+)(-cu\d+)-(ubuntu\d+\.\d+)((-e3)?-example|-e3|-sagemaker)?)",
        image,
    ).group(1)

    # replacing '_' by '/' to handle huggingface_<framework> case
    framework_path = framework.replace("_", "/")
    framework_version_path = os.path.join(
        dlc_path, framework_path, job_type, "docker", framework_version)
    if not os.path.exists(framework_version_path):
        framework_short_version = re.match(
            r"(\d+.\d+)", framework_version).group(1)
        framework_version_path = os.path.join(
            dlc_path, framework_path, job_type, "docker", framework_short_version)
    if not os.path.exists(os.path.join(framework_version_path, python_version)):
        # Use the pyX version as opposed to the pyXY version if pyXY path does not exist
        short_python_version = python_version[:3]

    # Check buildspec for cuda version
    buildspec = "buildspec.yml"
    if is_tf_version("1", image):
        buildspec = "buildspec-tf1.yml"

    image_tag_in_buildspec = False
    dockerfile_spec_abs_path = None
    buildspec_path = os.path.join(dlc_path, framework_path, buildspec)
    buildspec_def = Buildspec()
    buildspec_def.load(buildspec_path)

    for name, image_spec in buildspec_def["images"].items():
        if image_spec["device_type"] == "gpu" and image_spec["tag"] == image_tag:
            image_tag_in_buildspec = True
            dockerfile_spec_abs_path = os.path.join(
                os.path.dirname(
                    framework_version_path), image_spec["docker_file"].lstrip("docker/")
            )
            break
    try:
        assert image_tag_in_buildspec, f"Image tag {image_tag} not found in {buildspec_path}"
    except AssertionError as e:
        if not is_dlc_cicd_context():
            LOGGER.warn(
                f"{e} - not failing, as this is a(n) {os.getenv('BUILD_CONTEXT', 'empty')} build context.")
        else:
            raise

    image_properties_expected_in_dockerfile_path = [
        framework_short_version or framework_version,
        short_python_version or python_version,
        cuda_version,
    ]
    assert all(prop in dockerfile_spec_abs_path for prop in image_properties_expected_in_dockerfile_path), (
        f"Dockerfile location {dockerfile_spec_abs_path} does not contain all the image properties in "
        f"{image_properties_expected_in_dockerfile_path}"
    )

    assert os.path.exists(
        dockerfile_spec_abs_path), f"Cannot find dockerfile for {image} in {dockerfile_spec_abs_path}"


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


@pytest.mark.usefixtures("sagemaker")
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
    run_cmd_on_container(container_name, context,
                         "/usr/local/bin/testOSSCompliance /root")

    try:
        context.run(
            f"docker cp {container_name}:/root/{file} {os.path.join(local_repo_path, file)}")
    finally:
        context.run(f"docker rm -f {container_name}", hide=True)

    s3_resource = boto3.resource("s3")

    with open(os.path.join(local_repo_path, file)) as source_code_file:
        for line in source_code_file:
            name, version, url = line.split(" ")
            file_name = f"{name}_v{version}_source_code"
            s3_object_path = f"{THIRD_PARTY_SOURCE_CODE_BUCKET_PATH}/{file_name}.tar.gz"
            local_file_path = os.path.join(local_repo_path, file_name)

            for i in range(3):
                try:
                    if not os.path.isdir(local_file_path):
                        context.run(
                            f"git clone {url.rstrip()} {local_file_path}")
                        context.run(
                            f"tar -czvf {local_file_path}.tar.gz {local_file_path}")
                except Exception as e:
                    time.sleep(1)
                    if i == 2:
                        LOGGER.error(f"Unable to clone git repo. Error: {e}")
                        raise
                    continue
            try:
                if os.path.exists(f"{local_file_path}.tar.gz"):
                    LOGGER.info(f"Uploading package to s3 bucket: {line}")
                    s3_resource.Object(
                        THIRD_PARTY_SOURCE_CODE_BUCKET, s3_object_path).load()
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    try:
                        # using aws cli as using boto3 expects to upload folder by iterating through each file instead of entire folder.
                        context.run(
                            f"aws s3 cp {local_file_path}.tar.gz s3://{THIRD_PARTY_SOURCE_CODE_BUCKET}/{s3_object_path}"
                        )
                        object = s3_resource.Bucket(
                            THIRD_PARTY_SOURCE_CODE_BUCKET).Object(s3_object_path)
                        object.Acl().put(ACL="public-read")
                    except ClientError as e:
                        LOGGER.error(
                            f"Unable to upload source code to bucket {THIRD_PARTY_SOURCE_CODE_BUCKET}. Error: {e}"
                        )
                        raise
                else:
                    LOGGER.error(
                        f"Unable to check if source code is present on bucket {THIRD_PARTY_SOURCE_CODE_BUCKET}. Error: {e}"
                    )
                    raise
