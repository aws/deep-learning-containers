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
    is_dlc_cicd_context,
    run_cmd_on_container,
    start_container,
    stop_and_remove_container,
    get_repository_local_path,
    get_repository_and_tag_from_image_uri,
    get_python_version_from_image_uri,
    get_cuda_version_from_tag,
    construct_buildspec_path,
    is_tf_version,
    is_nightly_context,
    get_processor_from_image_uri,
    execute_env_variables_test,
    UL20_CPU_ARM64_US_WEST_2,
    UBUNTU_18_HPU_DLAMI_US_WEST_2,
    NEURON_UBUNTU_18_BASE_DLAMI_US_WEST_2,
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
@pytest.mark.canary("Run non-gpu tf serving version test regularly on production images")
def test_tf_serving_version_cpu(tensorflow_inference):
    """
    For non-huggingface non-GPU TF inference images, check that the tag version matches the version of TF serving
    in the container.

    Huggingface includes MMS and core TF, hence the versioning scheme is based off of the underlying tensorflow
    framework version, rather than the TF serving version.

    GPU inference images will be tested along side `test_framework_and_cuda_version_gpu` in order to be judicious
    about GPU resources. This test can run directly on the host, and thus does not require additional resources
    to be spun up.

    @param tensorflow_inference: ECR image URI
    """
    # Set local variable to clarify contents of fixture
    image = tensorflow_inference

    if "gpu" in image:
        pytest.skip(
            "GPU images will have their framework version tested in test_framework_and_cuda_version_gpu"
        )
    if "neuron" in image:
        pytest.skip(
            "Neuron images will have their framework version tested in test_framework_and_neuron_sdk_version"
        )

    _, tag_framework_version = get_framework_and_version_from_tag(image)

    image_repo_name, _ = get_repository_and_tag_from_image_uri(image)

    if re.fullmatch(r"(pr-|beta-|nightly-)?tensorflow-inference", image_repo_name) and Version(
        tag_framework_version
    ) == Version("2.6.3"):
        pytest.skip(
            "Skipping this test for TF 2.6.3 inference as the v2.6.3 version is already on production"
        )

    ctx = Context()
    container_name = get_container_name("tf-serving-version", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name, ctx, "tensorflow_model_server --version", executable="bash"
    )
    assert re.match(
        rf"TensorFlow ModelServer: {tag_framework_version}(\D+)?", output.stdout
    ), f"Cannot find model server version {tag_framework_version} in {output.stdout}"

    stop_and_remove_container(container_name, ctx)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
def test_tf_serving_api_version(tensorflow_inference):
    """
    For non-huggingface TF inference images, check that the tag version matches the version of TF serving api
    in the container.

    Huggingface includes MMS and core TF, hence the versioning scheme is based off of the underlying tensorflow
    framework version, rather than the TF serving version.

    @param tensorflow_inference: ECR image URI
    """
    # Set local variable to clarify contents of fixture
    image = tensorflow_inference

    if "gpu" in image:
        cmd = "pip show tensorflow-serving-api-gpu | grep Version"
    elif "cpu" in image:
        cmd = "pip show tensorflow-serving-api | grep Version"
    else:
        ValueError(
            "Test as of now only covers CPU and GPU type images. If required, please modify this test to accommodate the new image type!"
        )

    _, tag_framework_version = get_framework_and_version_from_tag(image)

    ctx = Context()
    container_name = get_container_name("tf-serving-api-version", image)
    start_container(container_name, image, ctx)
    try:
        output = run_cmd_on_container(container_name, ctx, cmd, executable="bash")
        str_version_from_output = ((str(output.stdout).split(" "))[1]).strip()
        assert (
            tag_framework_version == str_version_from_output
        ), f"Tensorflow serving API version is {str_version_from_output} while the Tensorflow version is {tag_framework_version}. Both don't match!"
    except Exception as e:
        LOGGER.error(f"Unable to execute command on container. Error: {e}")
        raise
    finally:
        stop_and_remove_container(container_name, ctx)


@pytest.mark.usefixtures("sagemaker", "huggingface")
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
            "GPU images will have their framework version tested in test_framework_and_cuda_version_gpu"
        )
    if "neuron" in image:
        pytest.skip(
            "Neuron images will have their framework version tested in test_framework_and_neuron_sdk_version"
        )
    image_repo_name, _ = get_repository_and_tag_from_image_uri(image)
    if re.fullmatch(r"(pr-|beta-|nightly-)?tensorflow-inference(-eia|-graviton)?", image_repo_name):
        pytest.skip(
            "Non-gpu tensorflow-inference images will be tested in test_tf_serving_version_cpu."
        )

    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)
    # Framework name may include huggingface
    if any([tested_framework.startswith(prefix) for prefix in ["huggingface_", "stabilityai_"]]):
        # Remove the prefix till first underscore
        tested_framework = "_".join(tested_framework.split("_")[1:])
    # Module name is torch
    if tested_framework == "pytorch":
        tested_framework = "torch"
    elif tested_framework == "autogluon":
        tested_framework = "autogluon.core"
    ctx = Context()
    container_name = get_container_name("framework-version", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name,
        ctx,
        f"import {tested_framework}; print({tested_framework}.__version__)",
        executable="python",
    ).stdout.strip()
    if is_canary_context():
        assert tag_framework_version in output
    else:
        if tested_framework == "autogluon.core":
            versions_map = {
                # container version -> autogluon version
                # '0.3.2': '0.3.1',
            }
            version_to_check = versions_map.get(tag_framework_version, tag_framework_version)
            assert output.startswith(version_to_check)
        # Habana v1.2 binary does not follow the X.Y.Z+cpu naming convention
        elif "habana" not in image_repo_name:
            if tested_framework == "torch" and Version(tag_framework_version) >= Version("1.10.0"):
                if is_nightly_context():
                    torch_version_pattern = r"{torch_version}(\+cpu|\.dev\d+)".format(
                        torch_version=tag_framework_version
                    )
                    assert re.fullmatch(torch_version_pattern, output), (
                        f"torch.__version__ = {output} does not match {torch_version_pattern}\n"
                        f"Please specify nightly framework version as X.Y.Z.devYYYYMMDD"
                    )
                else:
                    if (
                        Version(tag_framework_version) >= Version("2.0.0")
                        and "training" in image_repo_name
                    ):
                        cuda_output = run_cmd_on_container(
                            container_name,
                            ctx,
                            f"import {tested_framework}; print({tested_framework}.version.cuda)",
                            executable="python",
                        ).stdout.strip()
                        torch_version_pattern = r"{torch_version}".format(
                            torch_version=tag_framework_version
                        )
                        assert cuda_output == "None", f"cuda version has value: {cuda_output}"
                    else:
                        torch_version_pattern = r"{torch_version}(\+cpu)".format(
                            torch_version=tag_framework_version
                        )
                    assert re.fullmatch(torch_version_pattern, output), (
                        f"torch.__version__ = {output} does not match {torch_version_pattern}\n"
                        f"Please specify framework version as X.Y.Z+cpu"
                    )
        else:
            if "neuron" in image:
                assert tag_framework_version in output
            if all(_string in image for _string in ["pytorch", "habana"]) and any(
                _string in image
                for _string in ["synapseai1.3.0", "synapseai1.4.1", "synapseai1.5.0"]
            ):
                # Habana Pytorch version looks like 1.10.0a0+gitb488e78 for SynapseAI1.3 PT1.10.1 images
                pt_fw_version_pattern = r"(\d+(\.\d+){1,2}(-rc\d)?)((a0\+git\w{7}))"
                pt_fw_version_match = re.fullmatch(pt_fw_version_pattern, output)
                # This is desired for PT1.10.1 images
                assert (
                    tag_framework_version.rsplit(".", 1)[0]
                    == pt_fw_version_match.group(1).rsplit(".", 1)[0]
                )
            else:
                assert tag_framework_version == output
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

    tested_framework, neuron_tag_framework_version = get_neuron_framework_and_version_from_tag(
        image
    )

    # neuron tag is there in pytorch images for now. Once all frameworks have it, then this will
    # be removed
    if neuron_tag_framework_version is None:
        if tested_framework == "pytorch":
            assert neuron_tag_framework_version != None
        else:
            pytest.skip(msg="Neuron SDK tag is not there as part of image")

    # Framework name may include huggingface
    if tested_framework.startswith("huggingface_"):
        tested_framework = tested_framework[len("huggingface_") :]

    if tested_framework == "pytorch":
        if "training" in image or "neuronx" in image:
            tested_framework = "torch_neuronx"
        else:
            tested_framework = "torch_neuron"
    elif tested_framework == "tensorflow":
        if "neuronx" in image:
            tested_framework = "tensorflow_neuronx"
        else:
            tested_framework = "tensorflow_neuron"
    elif tested_framework == "mxnet":
        tested_framework = "mxnet"

    ctx = Context()

    container_name = get_container_name("framework-version-neuron", image)
    start_container(container_name, image, ctx)
    output = run_cmd_on_container(
        container_name,
        ctx,
        f"import {tested_framework}; print({tested_framework}.__version__)",
        executable="python",
    )

    if tested_framework == "mxnet":
        # TODO -For neuron the mx_neuron module does not support the __version__ yet and we
        # can get the version of only the base mxnet model. The base mxnet model just
        # has framework version and does not have the neuron semantic version yet. Till
        # the mx_neuron supports __version__ do the minimal check and not exact match
        _, tag_framework_version = get_framework_and_version_from_tag(image)
        assert tag_framework_version == output.stdout.strip()
    else:
        assert neuron_tag_framework_version == output.stdout.strip()
    stop_and_remove_container(container_name, ctx)


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
    tested_framework, tag_framework_version = get_framework_and_version_from_tag(image)

    image_repo_name, _ = get_repository_and_tag_from_image_uri(image)

    if re.fullmatch(r"(pr-|beta-|nightly-)?tensorflow-inference", image_repo_name) and Version(
        tag_framework_version
    ) == Version("2.6.3"):
        pytest.skip(
            "Skipping this test for TF 2.6.3 inference as the v2.6.3 version is already on production"
        )

    # Framework Version Check #
    # For tf inference containers, check TF model server version
    if re.fullmatch(r"(pr-|beta-|nightly-)?tensorflow-inference(-eia|-graviton)?", image_repo_name):
        cmd = f"tensorflow_model_server --version"
        output = ec2.execute_ec2_training_test(ec2_connection, image, cmd, executable="bash").stdout
        assert re.match(
            rf"TensorFlow ModelServer: {tag_framework_version}(\D+)?", output
        ), f"Cannot find model server version {tag_framework_version} in {output}"
    else:
        # Framework name may include huggingface
        if any(
            [tested_framework.startswith(prefix) for prefix in ["huggingface_", "stabilityai_"]]
        ):
            tested_framework = "_".join(tested_framework.split("_")[1:])
            # Replace the trcomp string as it is extracted from ECR repo name
            tested_framework = tested_framework.replace("_trcomp", "")
        # Framework name may include trcomp
        if "trcomp" in tested_framework:
            # Replace the trcomp string as it is extracted from ECR repo name
            tested_framework = tested_framework.replace("_trcomp", "")
        # Module name is "torch"
        if tested_framework == "pytorch":
            tested_framework = "torch"
        elif tested_framework == "autogluon":
            tested_framework = "autogluon.core"
        cmd = f"import {tested_framework}; print({tested_framework}.__version__)"
        output = ec2.execute_ec2_training_test(
            ec2_connection, image, cmd, executable="python"
        ).stdout.strip()
        if is_canary_context():
            assert tag_framework_version in output
        else:
            if tested_framework == "autogluon.core":
                # If tag and framework are not matching:
                # version_to_check = "0.3.1" if tag_framework_version == "0.3.2" else tag_framework_version
                # assert output.stdout.strip().startswith(version_to_check)
                pass
            elif tested_framework == "torch" and Version(tag_framework_version) >= Version(
                "1.10.0"
            ):
                if is_nightly_context():
                    torch_version_pattern = r"{torch_version}(\+cu\d+|\.dev\d+)".format(
                        torch_version=tag_framework_version
                    )
                    assert re.fullmatch(torch_version_pattern, output), (
                        f"torch.__version__ = {output} does not match {torch_version_pattern}\n"
                        f"Please specify nightly framework version as X.Y.Z.devYYYYMMDD"
                    )
                else:
                    if (
                        Version(tag_framework_version) >= Version("2.0.0")
                        and "training" in image_repo_name
                    ):
                        cuda_output = ec2.execute_ec2_training_test(
                            ec2_connection,
                            image,
                            'import torch; print(torch.version.cuda.replace(".", ""));',
                            executable="python",
                            container_name="PT2",
                        ).stdout.strip()
                        cuda_ver = get_cuda_version_from_tag(image)
                        torch_version_pattern = r"{torch_version}".format(
                            torch_version=tag_framework_version
                        )
                        assert (
                            output == tag_framework_version
                        ), f"torch.__version__ = {output} does not match {torch_version_pattern}\n"
                        assert (
                            cuda_ver == "cu" + cuda_output
                        ), f"torch.version.cuda {cuda_ver} doesn't match {cuda_output}"
                    else:
                        torch_version_pattern = r"{torch_version}(\+cu\d+)".format(
                            torch_version=tag_framework_version
                        )
                        assert re.fullmatch(torch_version_pattern, output), (
                            f"torch.__version__ = {output} does not match {torch_version_pattern}\n"
                            f"Please specify framework version as X.Y.Z+cuXXX"
                        )
            else:
                assert tag_framework_version == output

    # CUDA Version Check #
    cuda_version = re.search(r"-cu(\d+)-", image).group(1)

    # MXNet inference/HF tensorflow inference and Autogluon containers do not currently have nvcc in /usr/local/cuda/bin, so check symlink
    if (
        "mxnet-inference" in image
        or "autogluon" in image
        or "huggingface-tensorflow-inference" in image
    ):
        cuda_cmd = "readlink -f /usr/local/cuda"
    else:
        cuda_cmd = "nvcc --version"
    cuda_output = ec2.execute_ec2_training_test(
        ec2_connection, image, cuda_cmd, container_name="cuda_version_test"
    )

    # Ensure that cuda version in tag is in the container
    assert cuda_version in cuda_output.stdout.replace(".", "")


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

    python_version = get_python_version_from_image_uri(image).replace("py", "")
    python_version = int(python_version)

    if python_version >= 37:
        start_container(container_name, image, ctx)
        output = run_cmd_on_container(container_name, ctx, f"pip show {pip_package}", warn=True)

        if output.return_code == 0:
            pytest.fail(
                f"{pip_package} package exists in the DLC image {image} that has py{python_version} version which is greater than py36 version"
            )
        else:
            LOGGER.info(f"{pip_package} package does not exists in the DLC image {image}")
    else:
        pytest.skip(
            f"Skipping test for DLC image {image} that has py36 version as {pip_package} is not included in the python framework"
        )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
def test_pip_check(image):
    """
    Ensure there are no broken requirements on the containers by running "pip check"

    :param image: ECR image URI
    """

    ctx = Context()
    gpu_suffix = "-gpu" if "gpu" in image else ""
    allowed_exception_list = []

    # SageMaker Python SDK updated its pyyaml requirement to 6.0, which is incompatible with the
    # requirement from awscli. awscli only requires pyyaml for ecs/eks related invocations, while
    # pyyaml usage seems to be more fundamental in sagemaker. Therefore, we are ignoring awscli's
    # requirement in favor of sagemaker.
    allowed_awscli_exception = re.compile(
        r"^awscli \d+(\.\d+)* has requirement PyYAML<5\.5,>=3\.10, but you have pyyaml 6\.0.$"
    )
    allowed_exception_list.append(allowed_awscli_exception)

    # TF inference containers do not have core tensorflow installed by design. Allowing for this pip check error
    # to occur in order to catch other pip check issues that may be associated with TF inference
    # smclarify binaries have s3fs->aiobotocore dependency which uses older version of botocore. temporarily
    # allowing this to catch other issues
    allowed_tf_exception = re.compile(
        rf"^tensorflow-serving-api{gpu_suffix} \d\.\d+\.\d+ requires tensorflow(|{gpu_suffix}), which is not installed.$"
    )
    allowed_exception_list.append(allowed_tf_exception)

    allowed_smclarify_exception = re.compile(
        r"^aiobotocore \d+(\.\d+)* has requirement botocore<\d+(\.\d+)*,>=\d+(\.\d+)*, "
        r"but you have botocore \d+(\.\d+)*\.$"
    )
    allowed_exception_list.append(allowed_smclarify_exception)

    # The v0.22 version of tensorflow-io has a bug fixed in v0.23 https://github.com/tensorflow/io/releases/tag/v0.23.0
    allowed_habana_tf_exception = re.compile(
        rf"^tensorflow-io 0.22.0 requires tensorflow, which is not installed.$"
    )
    allowed_exception_list.append(allowed_habana_tf_exception)

    framework, framework_version = get_framework_and_version_from_tag(image)
    # The v0.21 version of tensorflow-io has a bug fixed in v0.23 https://github.com/tensorflow/io/releases/tag/v0.23.0

    tf263_io21_issue_framework_list = [
        "tensorflow",
        "huggingface_tensorflow",
        "huggingface_tensorflow_trcomp",
    ]
    if framework in tf263_io21_issue_framework_list or Version(framework_version) in SpecifierSet(
        ">=2.6.3,<2.7"
    ):
        allowed_tf263_exception = re.compile(
            rf"^tensorflow-io 0.21.0 requires tensorflow, which is not installed.$"
        )
        allowed_exception_list.append(allowed_tf263_exception)

    # TF2.9 sagemaker containers introduce tf-models-official which has a known bug where in it does not respect the
    # existing TF installation. https://github.com/tensorflow/models/issues/9267. This package in turn brings in
    # tensorflow-text. Skip checking these two packages as this is an upstream issue.
    if framework in ["tensorflow", "huggingface_tensorflow"] and Version(
        framework_version
    ) in SpecifierSet(">=2.9.1"):
        exception_strings = []
        models_versions = ["2.9.1", "2.9.2", "2.10.0", "2.11.0", "2.12.0"]
        for ex_ver in models_versions:
            exception_strings += [f"tf-models-official {ex_ver}".replace(".", "\.")]
        text_versions = ["2.9.0", "2.10.0", "2.11.0", "2.12.0"]
        for ex_ver in text_versions:
            exception_strings += [f"tensorflow-text {ex_ver}".replace(".", "\.")]
        allowed_tf_models_text_exception = re.compile(
            rf"^({'|'.join(exception_strings)}) requires tensorflow, which is not installed."
        )
        allowed_exception_list.append(allowed_tf_models_text_exception)

        allowed_tf_models_text_compatibility_exception = re.compile(
            rf"tf-models-official 2.9.2 has requirement tensorflow-text~=2.9.0, but you have tensorflow-text 2.10.0."
        )
        allowed_exception_list.append(allowed_tf_models_text_compatibility_exception)

    if "pytorch" in image and "trcomp" in image:
        allowed_exception_list.append(
            re.compile(r"torch-xla \d+(\.\d+)* requires absl-py, which is not installed.")
        )
        allowed_exception_list.append(
            re.compile(r"torch-xla \d+(\.\d+)* requires cloud-tpu-client, which is not installed.")
        )

    # Add null entrypoint to ensure command exits immediately
    output = ctx.run(f"docker run --entrypoint='' {image} pip check", hide=True, warn=True)
    if output.return_code != 0:
        if not (
            any(
                [
                    allowed_exception.findall(output.stdout)
                    for allowed_exception in allowed_exception_list
                ]
            )
        ):
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
        pytest.skip("Skipping Example Dockerfiles which are not explicitly tied to a cuda version")

    dlc_path = os.getcwd().split("/test/")[0]
    job_type = "training" if "training" in image else "inference"

    # Ensure that image has a supported framework
    framework, framework_version = get_framework_and_version_from_tag(image)

    # Get cuda, framework version, python version through regex
    cuda_version = re.search(r"-(cu\d+)-", image).group(1)

    framework_short_version = re.match(r"(\d+\.\d+)", framework_version).group(1)

    python_version = re.search(r"(py\d+)", image).group(1)
    short_python_version = None
    image_tag = re.search(
        r":(\d+(\.\d+){2}(-(transformers|diffusers)\d+(\.\d+){2})?-(gpu)-(py\d+)(-cu\d+)-(ubuntu\d+\.\d+)((-ec2)?-example|-ec2|-sagemaker-lite|-sagemaker-full|-sagemaker)?)",
        image,
    ).group(1)

    # replacing '_' by '/' to handle huggingface_<framework> case
    framework = framework.replace("_trcomp", "")
    framework_path = framework.replace("_", "/")
    framework_version_path = os.path.join(
        dlc_path, framework_path, job_type, "docker", framework_version
    )

    if not os.path.exists(framework_version_path):
        framework_version_path = os.path.join(
            dlc_path, framework_path, job_type, "docker", framework_short_version
        )

    if not os.path.exists(os.path.join(framework_version_path, python_version)):
        # Use the pyX version as opposed to the pyXY version if pyXY path does not exist
        short_python_version = python_version[:3]

    # Check buildspec for cuda version
    buildspec = "buildspec"
    if is_tf_version("1", image):
        buildspec = "buildspec-tf1"
    if "trcomp" in image:
        buildspec = "buildspec-trcomp"
    if "sagemaker-lite" in image:
        buildspec = "buildspec-sagemaker-lite"

    image_tag_in_buildspec = False
    dockerfile_spec_abs_path = None

    buildspec_path = construct_buildspec_path(
        dlc_path, framework_path, buildspec, framework_version, job_type
    )
    buildspec_def = Buildspec()
    buildspec_def.load(buildspec_path)

    for name, image_spec in buildspec_def["images"].items():
        if image_spec["device_type"] == "gpu" and image_spec["tag"] == image_tag:
            image_tag_in_buildspec = True
            dockerfile_spec_abs_path = os.path.join(
                os.path.dirname(framework_version_path), image_spec["docker_file"].lstrip("docker/")
            )
            break
    try:
        assert image_tag_in_buildspec, f"Image tag {image_tag} not found in {buildspec_path}"
    except AssertionError as e:
        if not is_dlc_cicd_context():
            LOGGER.warn(
                f"{e} - not failing, as this is a(n) {os.getenv('BUILD_CONTEXT', 'empty')} build context."
            )
        else:
            raise

    image_properties_expected_in_dockerfile_path = [
        framework_short_version or framework_version,
        short_python_version or python_version,
        cuda_version,
    ]
    assert all(
        prop in dockerfile_spec_abs_path for prop in image_properties_expected_in_dockerfile_path
    ), (
        f"Dockerfile location {dockerfile_spec_abs_path} does not contain all the image properties in "
        f"{image_properties_expected_in_dockerfile_path}"
    )

    assert os.path.exists(
        dockerfile_spec_abs_path
    ), f"Cannot find dockerfile for {image} in {dockerfile_spec_abs_path}"


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
@pytest.mark.skipif(
    not is_dlc_cicd_context(), reason="We need to test OSS compliance only on PRs and pipelines"
)
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
        context.run(
            f"docker cp {container_name}:/root/{file} {os.path.join(local_repo_path, file)}"
        )
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
                        context.run(f"git clone {url.rstrip()} {local_file_path}", hide=True)
                        context.run(f"tar -czvf {local_file_path}.tar.gz {local_file_path}")
                except Exception as e:
                    time.sleep(1)
                    if i == 2:
                        LOGGER.error(f"Unable to clone git repo. Error: {e}")
                        raise
                    continue
            try:
                if os.path.exists(f"{local_file_path}.tar.gz"):
                    LOGGER.info(f"Uploading package to s3 bucket: {line}")
                    s3_resource.Object(THIRD_PARTY_SOURCE_CODE_BUCKET, s3_object_path).load()
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    try:
                        # using aws cli as using boto3 expects to upload folder by iterating through each file instead of entire folder.
                        context.run(
                            f"aws s3 cp {local_file_path}.tar.gz s3://{THIRD_PARTY_SOURCE_CODE_BUCKET}/{s3_object_path}"
                        )
                        object = s3_resource.Bucket(THIRD_PARTY_SOURCE_CODE_BUCKET).Object(
                            s3_object_path
                        )
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


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
def test_pytorch_training_sm_env_variables(pytorch_training):
    env_vars = {"SAGEMAKER_TRAINING_MODULE": "sagemaker_pytorch_container.training:main"}
    container_name_prefix = "pt_training_sm_env"
    execute_env_variables_test(
        image_uri=pytorch_training,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
def test_pytorch_inference_sm_env_variables(pytorch_inference):
    env_vars = {"SAGEMAKER_SERVING_MODULE": "sagemaker_pytorch_serving_container.serving:main"}
    container_name_prefix = "pt_inference_sm_env"
    execute_env_variables_test(
        image_uri=pytorch_inference,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
def test_tensorflow_training_sm_env_variables(tensorflow_training):
    env_vars = {"SAGEMAKER_TRAINING_MODULE": "sagemaker_tensorflow_container.training:main"}
    container_name_prefix = "tf_training_sm_env"
    execute_env_variables_test(
        image_uri=tensorflow_training,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
def test_tensorflow_inference_sm_env_variables(tensorflow_inference):
    _, fw_version = get_framework_and_version_from_tag(tensorflow_inference)
    version_obj = Version(fw_version)
    tf_short_version = f"{version_obj.major}.{version_obj.minor}"
    env_vars = {"SAGEMAKER_TFS_VERSION": tf_short_version}
    container_name_prefix = "tf_inference_sm_env"
    execute_env_variables_test(
        image_uri=tensorflow_inference,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.model("N/A")
def test_mxnet_training_sm_env_variables(mxnet_training):
    env_vars = {"SAGEMAKER_TRAINING_MODULE": "sagemaker_mxnet_container.training:main"}
    container_name_prefix = "mx_training_sm_env"
    execute_env_variables_test(
        image_uri=mxnet_training,
        env_vars_to_test=env_vars,
        container_name_prefix=container_name_prefix,
    )
