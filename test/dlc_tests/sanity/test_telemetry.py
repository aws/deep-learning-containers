import time

import pytest

from packaging.version import Version
from packaging.specifiers import SpecifierSet

from test import test_utils
from test.test_utils import ec2 as ec2_utils


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_telemetry_instance_role_disabled_gpu(gpu, ec2_client, ec2_instance, ec2_connection):
    _run_instance_role_disabled(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c4.4xlarge"], indirect=True)
def test_telemetry_bad_instance_role_disabled_cpu(cpu, ec2_client, ec2_instance, ec2_connection, cpu_only, x86_compatible_only):
    _run_instance_role_disabled(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.AML2_CPU_ARM64_US_WEST_2], indirect=True)
def test_telemetry_bad_instance_role_disabled_graviton_cpu(cpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only):
    _run_instance_role_disabled(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("neuron")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skip("Feature doesn't exist on Neuron DLCs")
def test_telemetry_bad_instance_role_disabled_neuron(neuron, ec2_client, ec2_instance, ec2_connection):
    _run_instance_role_disabled(neuron, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["p3.2xlarge"], indirect=True)
def test_telemetry_instance_tag_success_gpu(gpu, ec2_client, ec2_instance, ec2_connection, non_huggingface_only, non_autogluon_only):
    _run_tag_success(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c4.4xlarge"], indirect=True)
def test_telemetry_instance_tag_success_cpu(cpu, ec2_client, ec2_instance, ec2_connection, cpu_only, non_huggingface_only, non_autogluon_only, x86_compatible_only):
    _run_tag_success(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.AML2_CPU_ARM64_US_WEST_2], indirect=True)
def test_telemetry_instance_tag_success_graviton_cpu(cpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only):
    _run_tag_success(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.processor("neuron")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skip("Feature doesn't exist on Neuron DLCs")
def test_telemetry_instance_tag_success_neuron(neuron, ec2_client, ec2_instance, ec2_connection, non_huggingface_only, non_autogluon_only):
    _run_tag_success(neuron, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.usefixtures("sagemaker", "huggingface")
@pytest.mark.model("N/A")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g4dn.xlarge"], indirect=True)
def test_telemetry_silent_failure(image, ec2_instance_type, ec2_connection):
    """
    This test takes a DLC, overwrites the /usr/local/bin/deep_learning_container.py script with a script that always
    generates an Exception. If the framework import still succeeds, the test succeeds. If the framework import fails,
    the telemetry trigger needs to be fixed.

    :param image: str DLC Image URI
    :param ec2_connection: fabric Connection object
    """
    framework, _ = test_utils.get_framework_and_version_from_tag(image, import_safe_name=True)
    device_type = test_utils.get_processor_from_image_uri(image)
    if framework == "autogluon" or device_type == "neuron":
        pytest.skip("Skip this test for DLCs where telemetry is not used.")

    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image)
    container_name = f"{repo_name}-telemetry-silent-failure"

    result = _run_import_or_tfs_test(image, ec2_connection, container_name)

    assert result.ok, (
        f"Failed to import {framework} in {image} due to telemetry failure. "
        f"Fix this by making telemetry failures silent when they occur."
    )


def _run_instance_role_disabled(image_uri, ec2_client, ec2_instance, ec2_connection):
    expected_tag_key = "aws-dlc-autogenerated-tag-do-not-delete"

    ec2_instance_id, _ = ec2_instance
    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    processor = test_utils.get_processor_from_image_uri(image_uri)

    container_name = f"{repo_name}-telemetry_bad_instance_role-ec2"

    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"

    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{docker_cmd} pull -q {image_uri}")

    preexisting_ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    if expected_tag_key in preexisting_ec2_instance_tags:
        ec2_client.remove_tags(Resources=[ec2_instance_id], Tags=[{"Key": expected_tag_key}])

    # Disable access to EC2 instance metadata
    ec2_connection.run(f"sudo route add -host 169.254.169.254 reject")

    _run_import_or_tfs_test(image_uri, ec2_connection, container_name)

    ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    assert expected_tag_key not in ec2_instance_tags, (
        f"{expected_tag_key} was applied as an instance tag."
        "EC2 create_tags went through even though it should not have"
    )


def _run_tag_success(image_uri, ec2_client, ec2_instance, ec2_connection):
    expected_tag_key = "aws-dlc-autogenerated-tag-do-not-delete"

    ec2_instance_id, _ = ec2_instance
    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, _ = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    processor = test_utils.get_processor_from_image_uri(image_uri)

    container_name = f"{repo_name}-telemetry_tag_instance_success-ec2"

    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"

    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"{docker_cmd} pull -q {image_uri}")

    preexisting_ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    if expected_tag_key in preexisting_ec2_instance_tags:
        ec2_client.remove_tags(Resources=[ec2_instance_id], Tags=[{"Key": expected_tag_key}])

    _run_import_or_tfs_test(image_uri, ec2_connection, container_name)

    ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    assert expected_tag_key in ec2_instance_tags, f"{expected_tag_key} was not applied as an instance tag"


def _run_import_or_tfs_test(image_uri, ec2_connection, container_name, warn=False):
    framework, _ = test_utils.get_framework_and_version_from_tag(image_uri, import_safe_name=True)
    job_type = test_utils.get_job_type_from_image(image_uri)
    processor = test_utils.get_processor_from_image_uri(image_uri)
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"

    if "tensorflow" in framework and job_type == "inference":
        model_name = "saved_model_half_plus_two"
        model_base_path = test_utils.get_tensorflow_model_base_path(image_uri)
        env_vars_list = test_utils.get_tensorflow_inference_environment_variables(model_name, model_base_path)
        env_vars = " ".join([f"-e {entry['name']}={entry['value']}" for entry in env_vars_list])
        inference_command = get_tensorflow_inference_command_tf27_above(image_uri, model_name)
        output = ec2_connection.run(
            f"{docker_cmd} run {env_vars} --name {container_name} -id {image_uri} {inference_command}", warn=True
        )
        time.sleep(5)
        if not warn:
            assert output.ok, f"TF Model Server Inference failed"
    else:
        ec2_connection.run(f"{docker_cmd} run --name {container_name} -id {image_uri} bash")
        output = ec2_connection.run(
            f"{docker_cmd} exec -i {container_name} python -c 'import {framework}; import time; time.sleep(5)'",
            warn=True
        )
        if not warn:
            assert output.ok, f"'import {framework}' failed"
    return output


def get_tensorflow_inference_command_tf27_above(image_uri, model_name):

    _, image_framework_version = test_utils.get_framework_and_version_from_tag(image_uri)
    if Version(image_framework_version) in SpecifierSet(">=2.7"):
        inference_command = test_utils.build_tensorflow_inference_command_tf27_and_above(model_name)
        inference_shell_command = f"sh -c '{inference_command}'"
        return inference_shell_command
    else:
        return ""
