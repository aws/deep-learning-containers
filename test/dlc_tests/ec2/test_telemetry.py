import time

import pytest

from test import test_utils
from test.test_utils import ec2 as ec2_utils
from test.test_utils import LOGGER
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from test.test_utils import TELEMETRY_REGION_MAPPING


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "huggingface", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5.8xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, x86_compatible_only
):
    _run_tag_failure_IMDSv1_disabled(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "huggingface", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, cpu_only, x86_compatible_only
):
    _run_tag_failure_IMDSv1_disabled(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_graviton_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only
):
    ec2_connection.run(f"sudo dnf update -y")
    ec2_connection.run(f"sudo dnf install -y net-tools")
    _run_tag_failure_IMDSv1_disabled(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_arm64_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, arm64_compatible_only
):
    ec2_connection.run(f"sudo dnf update -y")
    ec2_connection.run(f"sudo dnf install -y net-tools")
    _run_tag_failure_IMDSv1_disabled(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.8xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_graviton_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only
):
    ec2_connection.run(f"sudo dnf update -y")
    ec2_connection.run(f"sudo dnf install -y net-tools")
    _run_tag_failure_IMDSv1_disabled(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.8xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_arm64_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, arm64_compatible_only
):
    ec2_connection.run(f"sudo dnf update -y")
    ec2_connection.run(f"sudo dnf install -y net-tools")
    _run_tag_failure_IMDSv1_disabled(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("neuron")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skip("Feature doesn't exist on Neuron DLCs")
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_failure_neuron(neuron, ec2_client, ec2_instance, ec2_connection):
    _run_tag_failure_IMDSv1_disabled(neuron, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5.8xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_success_gpu(
    gpu,
    ec2_client,
    ec2_instance,
    ec2_connection,
    non_huggingface_only,
    non_autogluon_only,
    non_pytorch_trcomp_only,
    x86_compatible_only,
):
    _run_tag_success_IMDSv1(gpu, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
def test_telemetry_instance_tag_success_cpu(
    cpu,
    ec2_client,
    ec2_instance,
    ec2_connection,
    cpu_only,
    non_huggingface_only,
    non_autogluon_only,
    x86_compatible_only,
):
    _run_tag_success_IMDSv1(cpu, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_success_graviton_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only
):
    _run_tag_success_IMDSv1(gpu, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_success_arm64_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, arm64_compatible_only
):
    _run_tag_success_IMDSv1(gpu, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.8xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_success_graviton_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only
):
    _run_tag_success_IMDSv1(cpu, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.8xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_success_arm64_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, arm64_compatible_only
):
    _run_tag_success_IMDSv1(cpu, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("neuron")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skip("Feature doesn't exist on Neuron DLCs")
@pytest.mark.timeout(1200)
def test_telemetry_instance_tag_success_neuron(
    neuron, ec2_client, ec2_instance, ec2_connection, non_huggingface_only, non_autogluon_only
):
    _run_tag_success_IMDSv1(neuron, ec2_client, ec2_instance, ec2_connection)
    _run_tag_success_IMDSv2_hop_limit_2(neuron, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5.8xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_gpu(
    gpu,
    ec2_client,
    ec2_instance,
    ec2_connection,
    non_huggingface_only,
    non_autogluon_only,
    x86_compatible_only,
):
    _run_s3_query_bucket_success(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_cpu(
    cpu,
    ec2_client,
    ec2_instance,
    ec2_connection,
    cpu_only,
    non_huggingface_only,
    non_autogluon_only,
    x86_compatible_only,
):
    _run_s3_query_bucket_success(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_graviton_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only
):
    _run_s3_query_bucket_success(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("gpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["g5g.2xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_arm64_gpu(
    gpu, ec2_client, ec2_instance, ec2_connection, arm64_compatible_only
):
    _run_s3_query_bucket_success(gpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.8xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_graviton_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, graviton_compatible_only
):
    _run_s3_query_bucket_success(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("cpu")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["c6g.8xlarge"], indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_arm64_cpu(
    cpu, ec2_client, ec2_instance, ec2_connection, arm64_compatible_only
):
    _run_s3_query_bucket_success(cpu, ec2_client, ec2_instance, ec2_connection)


@pytest.mark.flaky(reruns=2)
@pytest.mark.usefixtures("sagemaker", "telemetry")
@pytest.mark.model("N/A")
@pytest.mark.processor("neuron")
@pytest.mark.integration("telemetry")
@pytest.mark.parametrize("ec2_instance_type", ["inf1.xlarge"], indirect=True)
@pytest.mark.skip("Feature doesn't exist on Neuron DLCs")
@pytest.mark.timeout(1200)
def test_telemetry_s3_query_bucket_success_neuron(
    neuron, ec2_client, ec2_instance, ec2_connection, non_huggingface_only, non_autogluon_only
):
    _run_s3_query_bucket_success(neuron, ec2_client, ec2_instance, ec2_connection)


def _run_s3_query_bucket_success(image_uri, ec2_client, ec2_instance, ec2_connection):
    ec2_instance_id, _ = ec2_instance
    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    framework, framework_version = test_utils.get_framework_and_version_from_tag(image_uri)
    framework = framework.replace("stabilityai_", "")
    framework = framework.replace("_trcomp", "")
    job_type = test_utils.get_job_type_from_image(image_uri)
    container_type = test_utils.get_job_type_from_image(image_uri)
    container_name = f"{repo_name}-telemetry_s3_query_success-ec2"

    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ## For big images like trcomp, the ec2_connection.run command stops listening and the code hangs here.
    ## Hence, avoiding the use of -q to let the connection remain active.
    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    actual_output = invoke_telemetry_call(
        image_uri, container_name, framework, job_type, ec2_connection, test_mode="1"
    )

    py_version = (
        ec2_connection.run(
            f"docker exec -i {container_name} /bin/bash -c 'python --version'", warn=True
        )
        .stdout.strip("\n")
        .split(" ")[1]
    )

    expected_s3_url = (
        "https://aws-deep-learning-containers-{0}.s3.{1}.amazonaws.com"
        "/dlc-containers-{2}.txt?x-instance-id={2}&x-framework={3}&x-framework_version={4}&x-py_version={5}".format(
            TELEMETRY_REGION_MAPPING[image_region],
            image_region,
            ec2_instance_id,
            framework,
            framework_version,
            py_version,
        )
    )

    if (
        framework == "pytorch"
        and Version(framework_version) in SpecifierSet(">=2.0,<2.4")
        and container_type == "training"
    ):
        expected_s3_url += "&x-img_type=training&x-pkg_type=conda"
    else:
        expected_s3_url += f"&x-container_type={container_type}"

    assert expected_s3_url == actual_output, f"S3 telemetry is not working"


def _run_tag_failure_IMDSv1_disabled(image_uri, ec2_client, ec2_instance, ec2_connection):
    """
    Disable IMDSv1 on EC2 instance and try to add a tag in it, it should not get added
    """
    LOGGER.info(f"starting _run_tag_failure_IMDSv1_disabled with {image_uri}")
    expected_tag_key = "aws-dlc-autogenerated-tag-do-not-delete"

    ec2_instance_id, _ = ec2_instance
    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    framework, _ = test_utils.get_framework_and_version_from_tag(image_uri)
    job_type = test_utils.get_job_type_from_image(image_uri)

    container_name = f"{repo_name}-telemetry_instance_tag_failure-ec2"

    LOGGER.info(f"_run_tag_failure_IMDSv1_disabled pulling: {image_uri}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ## For big images like trcomp, the ec2_connection.run command stops listening and the code hangs here.
    ## Hence, avoiding the use of -q to let the connection remain active.
    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    LOGGER.info(f"_run_tag_failure_IMDSv1_disabled, {image_uri} get_ec2_instance_tags")
    preexisting_ec2_instance_tags = ec2_utils.get_ec2_instance_tags(
        ec2_instance_id, ec2_client=ec2_client
    )
    if expected_tag_key in preexisting_ec2_instance_tags:
        ec2_client.delete_tags(Resources=[ec2_instance_id], Tags=[{"Key": expected_tag_key}])

    # Disable access to EC2 instance metadata
    ec2_connection.run(f"sudo dnf install -y net-tools")
    ec2_connection.run(f"sudo route add -host 169.254.169.254 reject")

    invoke_telemetry_call(image_uri, container_name, framework, job_type, ec2_connection)

    LOGGER.info(f"_run_tag_failure_IMDSv1_disabled, {image_uri} starting get_ec2_instance_tags")
    ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    assert expected_tag_key not in ec2_instance_tags, (
        f"{expected_tag_key} was applied as an instance tag."
        "EC2 create_tags went through even though it should not have"
    )
    # Enable access to EC2 instance metadata, so other tests can be run on same EC2 instance
    ec2_connection.run(f"sudo route del -host 169.254.169.254 reject")


def _run_tag_success_IMDSv1(image_uri, ec2_client, ec2_instance, ec2_connection):
    """
    Try to add a tag on EC2 instance, it should get added as IMDSv1 is enabled by default
    """
    LOGGER.info(f"starting _run_tag_success_IMDSv1 with {image_uri}")
    expected_tag_key = "aws-dlc-autogenerated-tag-do-not-delete"

    ec2_instance_id, _ = ec2_instance
    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    framework, _ = test_utils.get_framework_and_version_from_tag(image_uri)
    job_type = test_utils.get_job_type_from_image(image_uri)

    container_name = f"{repo_name}-telemetry_tag_instance_success-ec2-IMDSv1"

    LOGGER.info(f"_run_tag_success_IMDSv1 pulling: {image_uri}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    LOGGER.info(f"_run_tag_success_IMDSv1, {image_uri} starting get_ec2_instance_tags")
    preexisting_ec2_instance_tags = ec2_utils.get_ec2_instance_tags(
        ec2_instance_id, ec2_client=ec2_client
    )
    LOGGER.info(
        f"_run_tag_success_IMDSv1, preexisting_ec2_instance_tags: {preexisting_ec2_instance_tags}"
    )
    if expected_tag_key in preexisting_ec2_instance_tags:
        ec2_client.delete_tags(Resources=[ec2_instance_id], Tags=[{"Key": expected_tag_key}])

    ec2_utils.enforce_IMDSv1(ec2_instance_id)

    invoke_telemetry_call(image_uri, container_name, framework, job_type, ec2_connection)

    LOGGER.info(f"_run_tag_success_IMDSv1, {image_uri} starting get_ec2_instance_tags")
    ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    LOGGER.info(f"ec2_instance_tags: {ec2_instance_tags}")
    assert (
        expected_tag_key in ec2_instance_tags
    ), f"{expected_tag_key} was not applied as an instance tag"

    # Change instance state back to IMDSv2 enabled with hop limit to 2
    ec2_utils.enforce_IMDSv2(ec2_instance_id, hop_limit=2)


# If hop limit on EC2 instance is 2, then IMDSv2 works as to get token IMDSv2 needs more than 1 hop
def _run_tag_success_IMDSv2_hop_limit_2(image_uri, ec2_client, ec2_instance, ec2_connection):
    """
    Try to add a tag on EC2 instance, it should get added as IMDSv2 is enabled due to hop limit 2
    """
    LOGGER.info(f"starting _run_tag_success_IMDSv2_hop_limit_2 with {image_uri}")
    expected_tag_key = "aws-dlc-autogenerated-tag-do-not-delete"

    ec2_instance_id, _ = ec2_instance
    account_id = test_utils.get_account_id_from_image_uri(image_uri)
    image_region = test_utils.get_region_from_image_uri(image_uri)
    repo_name, image_tag = test_utils.get_repository_and_tag_from_image_uri(image_uri)
    framework, _ = test_utils.get_framework_and_version_from_tag(image_uri)
    job_type = test_utils.get_job_type_from_image(image_uri)

    container_name = f"{repo_name}-telemetry_tag_instance_success-ec2-IMDSv2"

    LOGGER.info(f"_run_tag_success_IMDSv2_hop_limit_2 pulling: {image_uri}")
    test_utils.login_to_ecr_registry(ec2_connection, account_id, image_region)
    ec2_connection.run(f"docker pull {image_uri}", hide="out")

    LOGGER.info(f"_run_tag_success_IMDSv2_hop_limit_2, {image_uri} starting get_ec2_instance_tags")
    preexisting_ec2_instance_tags = ec2_utils.get_ec2_instance_tags(
        ec2_instance_id, ec2_client=ec2_client
    )
    LOGGER.info(
        f"_run_tag_success_IMDSv2_hop_limit_2, preexisting_ec2_instance_tags: {preexisting_ec2_instance_tags}"
    )
    if expected_tag_key in preexisting_ec2_instance_tags:
        ec2_client.delete_tags(Resources=[ec2_instance_id], Tags=[{"Key": expected_tag_key}])

    invoke_telemetry_call(image_uri, container_name, framework, job_type, ec2_connection)

    ec2_instance_tags = ec2_utils.get_ec2_instance_tags(ec2_instance_id, ec2_client=ec2_client)
    LOGGER.info(f"ec2_instance_tags: {ec2_instance_tags}")
    assert (
        expected_tag_key in ec2_instance_tags
    ), f"{expected_tag_key} was not applied as an instance tag"


def invoke_telemetry_call(
    image_uri, container_name, framework, job_type, ec2_connection, test_mode=None
):
    """
    Run import framework command inside docker container
    """
    output = None
    processor = test_utils.get_processor_from_image_uri(image_uri)
    docker_runtime = "--runtime=nvidia --gpus all" if processor == "gpu" else ""
    if "tensorflow" in framework and job_type == "inference":
        model_name = "saved_model_half_plus_two"
        model_base_path = test_utils.get_tensorflow_model_base_path(image_uri)
        env_vars_list = test_utils.get_tensorflow_inference_environment_variables(
            model_name, model_base_path
        )
        env_vars = " ".join([f"-e {entry['name']}={entry['value']}" for entry in env_vars_list])
        inference_command = get_tensorflow_inference_command_tf27_above(image_uri, model_name)
        if test_mode:
            ec2_connection.run(
                f"docker run {docker_runtime} {env_vars} -e TEST_MODE={test_mode} --name {container_name} -id {image_uri}  {inference_command}"
            )
            time.sleep(30)
            output = ec2_connection.run(
                f"docker exec -i {container_name} /bin/bash -c 'cat /tmp/test_request.txt'"
            ).stdout.strip("\n")
        else:
            ec2_connection.run(
                f"docker run {docker_runtime} {env_vars} --name {container_name} -id {image_uri} {inference_command}"
            )
            time.sleep(30)
    else:
        framework_to_import = (
            framework.replace("huggingface_", "").replace("_trcomp", "").replace("stabilityai_", "")
        )
        framework_to_import = "torch" if framework_to_import == "pytorch" else framework_to_import
        if test_mode:
            ec2_connection.run(
                f"docker run {docker_runtime} -e TEST_MODE={test_mode} --name {container_name} -id {image_uri} bash"
            )
            ec2_connection.run(
                f"docker exec -i -e TEST_MODE={test_mode} {container_name} python -c 'import {framework_to_import}; import time; time.sleep(30);'"
            )
            output = ec2_connection.run(
                f"docker exec -i {container_name} /bin/bash -c 'cat /tmp/test_request.txt'"
            ).stdout.strip("\n")
        else:
            ec2_connection.run(
                f"docker run {docker_runtime} --name {container_name} -id {image_uri} bash"
            )
            output = ec2_connection.run(
                f"docker exec -i {container_name} python -c 'import {framework_to_import}; import time; time.sleep(30)'"
            )
            assert (
                output.ok
            ), f"'import {framework_to_import}' fails when credentials not configured"
        time.sleep(30)
    return output


def get_tensorflow_inference_command_tf27_above(image_uri, model_name):
    _, image_framework_version = test_utils.get_framework_and_version_from_tag(image_uri)
    if Version(image_framework_version) in SpecifierSet(">=2.7"):
        entrypoint = "/usr/bin/tf_serving_entrypoint.sh"
        if "neuron" in image_uri:
            entrypoint = "/usr/local/bin/entrypoint.sh"
        inference_command = test_utils.build_tensorflow_inference_command_tf27_and_above(
            model_name, entrypoint
        )
        inference_shell_command = f"sh -c '{inference_command}'"
        return inference_shell_command
    else:
        return ""
