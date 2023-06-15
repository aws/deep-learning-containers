import datetime
import os
import subprocess
import random
import re

from time import sleep

import boto3
import invoke

from botocore.config import Config
from invoke.context import Context
from invoke import exceptions
from junit_xml import TestSuite, TestCase

from test_utils import ec2 as ec2_utils
from test_utils import metrics as metrics_utils
from test_utils import (
    destroy_ssh_keypair,
    generate_ssh_keypair,
    get_framework_and_version_from_tag,
    get_job_type_from_image,
    is_pr_context,
    SAGEMAKER_EXECUTION_REGIONS,
    SAGEMAKER_NEURON_EXECUTION_REGIONS,
    SAGEMAKER_NEURONX_EXECUTION_REGIONS,
    UBUNTU_18_BASE_DLAMI_US_EAST_1,
    UBUNTU_18_BASE_DLAMI_US_WEST_2,
    UL20_CPU_ARM64_US_EAST_1,
    UL20_CPU_ARM64_US_WEST_2,
    SAGEMAKER_LOCAL_TEST_TYPE,
    SAGEMAKER_REMOTE_TEST_TYPE,
    UBUNTU_HOME_DIR,
    DEFAULT_REGION,
    is_nightly_context,
)
from test_utils.pytest_cache import PytestCache


class DLCSageMakerRemoteTestFailure(Exception):
    pass


class DLCSageMakerLocalTestFailure(Exception):
    pass


def is_test_job_efa_dedicated():
    # In both CI Sagemaker Test CB jobs and DLC Build Pipeline Actions that run SM EFA tests, the env variable
    # "EFA_DEDICATED=True" must be configured so that those Actions only run the EFA tests.
    # This is change from the previous system where only setting env variable "DISABLE_EFA_TESTS=False" would enable
    # regular SM tests as well as SM EFA tests.
    return os.getenv("EFA_DEDICATED", "False").lower() == "true"


def assign_sagemaker_remote_job_instance_type(image):
    if "graviton" in image:
        return "ml.c6g.2xlarge"
    elif "neuronx" in image or "training-neuron" in image:
        return "ml.trn1.2xlarge"
    elif "inference-neuron" in image:
        return "ml.inf1.xlarge"
    elif "gpu" in image:
        return "ml.p3.8xlarge"
    elif "tensorflow" in image:
        return "ml.c4.4xlarge"
    else:
        return "ml.c4.8xlarge"


def assign_sagemaker_local_job_instance_type(image):
    if "graviton" in image:
        return "c6g.2xlarge"
    elif "tensorflow" in image and "inference" in image and "gpu" in image:
        return "g4dn.xlarge"
    elif "autogluon" in image and "gpu" in image:
        return "p3.2xlarge"
    elif "trcomp" in image:
        return "p3.2xlarge"
    return "p3.8xlarge" if "gpu" in image else "c5.18xlarge"


def assign_sagemaker_local_test_ami(image, region):
    """
    Helper function to get the needed AMI for launching the image.
    Needed to support Graviton(ARM) images
    """
    if "graviton" in image:
        if region == "us-east-1":
            return UL20_CPU_ARM64_US_EAST_1
        else:
            return UL20_CPU_ARM64_US_WEST_2
    else:
        if region == "us-east-1":
            return UBUNTU_18_BASE_DLAMI_US_EAST_1
        else:
            return UBUNTU_18_BASE_DLAMI_US_WEST_2


def launch_sagemaker_local_ec2_instance(image, ami_id, ec2_key_name, region):
    """
    Launch Ec2 instance for running sagemaker local tests
    :param image: str
    :param ami_id: str
    :param ec2_key_name: str
    :param region: str
    :return: str, str
    """
    instance_type = assign_sagemaker_local_job_instance_type(image)
    instance_name = image.split("/")[-1]
    instance = ec2_utils.launch_instance(
        ami_id,
        region=region,
        ec2_key_name=ec2_key_name,
        instance_type=instance_type,
        # EIA does not have SM Local test
        ei_accelerator_type=None,
        user_data=None,
        iam_instance_profile_name=ec2_utils.EC2_INSTANCE_ROLE_NAME,
        instance_name=f"sm-local-{instance_name}",
    )
    instance_id = instance["InstanceId"]
    public_ip_address = ec2_utils.get_public_ip(instance_id, region=region)
    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(
        instance_id, system_status="ok", instance_status="ok", region=region
    )
    return instance_id, public_ip_address


def generate_sagemaker_pytest_cmd(image, sagemaker_test_type):
    """
    Parses the image ECR url and returns appropriate pytest command
    :param image: ECR url of image
    :param sagemaker_test_type: local or remote test type
    :return: <tuple> pytest command to be run, path where it should be executed, image tag
    """
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    print("image name {}".format(image))
    sm_remote_docker_base_name, tag = image.split("/")[1].split(":")
    sm_local_docker_repo_uri = image.split(":")[0]

    # Assign instance type
    instance_type = assign_sagemaker_remote_job_instance_type(image)

    # Get path to test directory
    find_path = sm_remote_docker_base_name.split("-")

    # NOTE: We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework, framework_version = get_framework_and_version_from_tag(image)
    framework_major_version = framework_version.split(".")[0]
    job_type = get_job_type_from_image(image)
    framework = framework.replace("_trcomp", "").replace("stabilityai_", "")
    path = os.path.join("test", "sagemaker_tests", framework, job_type)
    aws_id_arg = "--aws-id"
    docker_base_arg = "--docker-base-name"
    instance_type_arg = "--instance-type"
    accelerator_type_arg = "--accelerator-type"
    framework_version_arg = "--framework-version"
    eia_arg = "ml.eia1.large"
    processor = (
        "neuronx"
        if "neuronx" in image
        else "neuron"
        if "neuron" in image
        else "gpu"
        if "gpu" in image
        else "eia"
        if "eia" in image
        else "cpu"
    )
    py_version = re.search(r"py\d+", tag).group()
    sm_local_py_version = (
        "37"
        if py_version == "py37"
        else "38"
        if py_version == "py38"
        else "39"
        if py_version == "py39"
        else "310"
        if py_version == "py310"
        else "2"
        if py_version == "py27"
        else "3"
    )
    if framework == "tensorflow" and job_type == "inference":
        # Tf Inference tests have an additional sub directory with test
        integration_path = os.path.join("test", "integration", sagemaker_test_type)
    else:
        integration_path = os.path.join("integration", sagemaker_test_type)

    # Conditions for modifying tensorflow SageMaker pytest commands
    if framework == "tensorflow" and sagemaker_test_type == SAGEMAKER_REMOTE_TEST_TYPE:
        if job_type == "inference":
            aws_id_arg = "--registry"
            docker_base_arg = "--repo"
            instance_type_arg = "--instance-types"
            framework_version_arg = "--versions"
            integration_path = (
                os.path.join(integration_path, "test_tfs.py")
                if processor != "eia"
                else os.path.join(integration_path, "test_ei.py")
            )

    if framework == "tensorflow" and job_type == "training":
        aws_id_arg = "--account-id"

    test_report = os.path.join(os.getcwd(), "test", f"{job_type}_{tag}.xml")
    local_test_report = os.path.join(UBUNTU_HOME_DIR, "test", f"{job_type}_{tag}_sm_local.xml")

    # In both CI Sagemaker Test CB jobs and DLC Build Pipeline Actions that run SM EFA tests, the env variable
    # "EFA_DEDICATED=True" must be configured so that those Actions only run the EFA tests.
    # This is change from the previous system where only setting env variable "DISABLE_EFA_TESTS=False" would enable
    # regular SM tests as well as SM EFA tests.
    efa_flag = "--efa" if is_test_job_efa_dedicated() else '-m "not efa"'

    region_list = SAGEMAKER_EXECUTION_REGIONS
    if "neuronx" in image:
        region_list = SAGEMAKER_NEURONX_EXECUTION_REGIONS
    elif "neuron" in image:
        region_list = SAGEMAKER_NEURON_EXECUTION_REGIONS

    region_list_str = ",".join(region_list)
    sagemaker_regions_list = f"--sagemaker-regions {region_list_str}"

    remote_pytest_cmd = (
        f"pytest -rA {integration_path} --region {region} --processor {processor} {docker_base_arg} "
        f"{sm_remote_docker_base_name} --tag {tag} {framework_version_arg} {framework_version} "
        f"{aws_id_arg} {account_id} {instance_type_arg} {instance_type} {efa_flag} {sagemaker_regions_list} --junitxml {test_report}"
    )

    if processor == "eia":
        remote_pytest_cmd += f" {accelerator_type_arg} {eia_arg}"

    local_pytest_cmd = (
        f"pytest -s -v {integration_path} {docker_base_arg} "
        f"{sm_local_docker_repo_uri} --tag {tag} --framework-version {framework_version} "
        f"--processor {processor} {aws_id_arg} {account_id} --junitxml {local_test_report}"
    )

    if framework == "tensorflow" and job_type != "inference":
        local_pytest_cmd = (
            f"{local_pytest_cmd} --py-version {sm_local_py_version} --region {region}"
        )
    if framework == "tensorflow" and job_type == "training":
        path = os.path.join(os.path.dirname(path), f"{framework}{framework_major_version}_training")
    if "huggingface" in framework and job_type == "inference":
        path = os.path.join("test", "sagemaker_tests", "huggingface", "inference")
    if "trcomp" in framework:
        path = os.path.join(
            "test", "sagemaker_tests", framework.replace("-trcomp", ""), f"{job_type}"
        )

    return (
        remote_pytest_cmd
        if sagemaker_test_type == SAGEMAKER_REMOTE_TEST_TYPE
        else local_pytest_cmd,
        path,
        tag,
        job_type,
    )


def execute_local_tests(image, pytest_cache_params):
    """
    Run the sagemaker local tests in ec2 instance for the image
    :param image: ECR url
    :param pytest_cache_params: parameters required for :param pytest_cache_util
    :return: True if test execution was successful, else False
    """
    test_success = False
    account_id = os.getenv("ACCOUNT_ID", boto3.client("sts").get_caller_identity()["Account"])
    pytest_cache_util = PytestCache(boto3.client("s3"), account_id)
    ec2_client = boto3.client(
        "ec2", config=Config(retries={"max_attempts": 10}), region_name=DEFAULT_REGION
    )
    pytest_command, path, tag, job_type = generate_sagemaker_pytest_cmd(
        image, SAGEMAKER_LOCAL_TEST_TYPE
    )
    pytest_command += " --last-failed --last-failed-no-failures all "
    print(pytest_command)
    framework, _ = get_framework_and_version_from_tag(image)
    framework = framework.replace("_trcomp", "")
    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    ec2_key_name = f"{job_type}_{tag}_sagemaker_{random.randint(1, 1000)}"
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    ec2_ami_id = assign_sagemaker_local_test_ami(image, region)
    sm_tests_tar_name = "sagemaker_tests.tar.gz"
    ec2_test_report_path = os.path.join(UBUNTU_HOME_DIR, "test", f"{job_type}_{tag}_sm_local.xml")
    instance_id = ""
    ec2_conn = None
    try:
        key_file = generate_ssh_keypair(ec2_client, ec2_key_name)
        print(f"Launching new Instance for image: {image}")
        instance_id, ip_address = launch_sagemaker_local_ec2_instance(
            image,
            ec2_ami_id,
            ec2_key_name,
            region,
        )
        ec2_conn = ec2_utils.get_ec2_fabric_connection(instance_id, key_file, region)
        ec2_conn.put(sm_tests_tar_name, f"{UBUNTU_HOME_DIR}")
        ec2_utils.install_python_in_instance(ec2_conn, python_version="3.9")
        ec2_conn.run(f"$(aws ecr get-login --no-include-email --region {region})")
        try:
            ec2_conn.run(f"docker pull {image}", timeout=600)
        except invoke.exceptions.CommandTimedOut as e:
            output = ec2_conn.run(
                f"docker images {image} --format '{{.Repository}}:{{.Tag}}'"
            ).stdout.strip("\n")
            if output != image:
                raise DLCSageMakerLocalTestFailure(
                    f"Image pull for {image} failed.\ndocker images output = {output}"
                ) from e
        ec2_conn.run(f"tar -xzf {sm_tests_tar_name}")
        with ec2_conn.cd(path):
            ec2_conn.run(f"pip install -r requirements.txt")
            pytest_cache_util.download_pytest_cache_from_s3_to_ec2(
                ec2_conn, path, **pytest_cache_params
            )
            # Workaround for mxnet cpu training images as test distributed
            # causes an issue with fabric ec2_connection
            if framework == "mxnet" and job_type == "training" and "cpu" in image:
                try:
                    ec2_conn.run(pytest_command, timeout=1000, warn=True)
                except exceptions.CommandTimedOut as exc:
                    print(f"Ec2 connection timed out for {image}, {exc}")
                finally:
                    print(f"Downloading Test reports for image: {image}")
                    ec2_conn.close()
                    ec2_conn_new = ec2_utils.get_ec2_fabric_connection(
                        instance_id, key_file, region
                    )
                    ec2_conn_new.get(
                        ec2_test_report_path, os.path.join("test", f"{job_type}_{tag}_sm_local.xml")
                    )
                    output = subprocess.check_output(
                        f"cat test/{job_type}_{tag}_sm_local.xml",
                        shell=True,
                        executable="/bin/bash",
                    )
                    pytest_cache_util.upload_pytest_cache_from_ec2_to_s3(
                        ec2_conn_new, path, **pytest_cache_params
                    )
                    if 'failures="0"' not in str(output):
                        if is_nightly_context():
                            print(f"\nSuppressed Failed Nightly Sagemaker Local Tests")
                        else:
                            raise ValueError(f"Sagemaker Local tests failed for {image}")
            else:
                res = ec2_conn.run(pytest_command, warn=True)
                print(f"Downloading Test reports for image: {image}")
                ec2_conn.get(
                    ec2_test_report_path, os.path.join("test", f"{job_type}_{tag}_sm_local.xml")
                )
                if res.failed:
                    if is_nightly_context():
                        print(f"Suppressed Failed Nightly Sagemaker Tests")
                        print(f"{pytest_command} failed with error code: {res.return_code}\n")
                        print(f"Traceback:\n{res.stderr}")
                    else:
                        raise DLCSageMakerLocalTestFailure(
                            f"{pytest_command} failed with error code: {res.return_code}\n"
                            f"Traceback:\n{res.stdout}"
                        )
        test_success = True
    except Exception as e:
        print(f"{type(e)} thrown : {str(e)}")
    finally:
        if ec2_conn:
            with ec2_conn.cd(path):
                pytest_cache_util.upload_pytest_cache_from_ec2_to_s3(
                    ec2_conn, path, **pytest_cache_params
                )
        if instance_id:
            print(f"Terminating Instances for image: {image}")
            ec2_utils.terminate_instance(instance_id, region)

        if ec2_client and ec2_key_name:
            print(f"Destroying ssh Key_pair for image: {image}")
            destroy_ssh_keypair(ec2_client, ec2_key_name)

    return test_success


def execute_sagemaker_remote_tests(process_index, image, global_pytest_cache, pytest_cache_params):
    """
    Run pytest in a virtual env for a particular image. Creates a custom directory for each thread for pytest cache file.
    Stores pytest cache in a shared dict.
    Expected to run via multiprocessing
    :param process_index - id for process. Used to create a custom cache dir
    :param image - ECR url
    :param global_pytest_cache - shared Manager().dict() for cache merging
    :param pytest_cache_params - parameters required for s3 file path building
    """
    account_id = os.getenv("ACCOUNT_ID", boto3.client("sts").get_caller_identity()["Account"])
    pytest_cache_util = PytestCache(boto3.client("s3"), account_id)
    pytest_command, path, tag, job_type = generate_sagemaker_pytest_cmd(
        image, SAGEMAKER_REMOTE_TEST_TYPE
    )
    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            pytest_cache_util.download_pytest_cache_from_s3_to_local(
                path, **pytest_cache_params, custom_cache_directory=str(process_index)
            )
            # adding -o cache_dir with a custom directory name
            pytest_command += f" -o cache_dir={os.path.join(str(process_index), '.pytest_cache')}"
            res = context.run(pytest_command, warn=True)
            metrics_utils.send_test_result_metrics(res.return_code)
            cache_json = pytest_cache_util.convert_pytest_cache_file_to_json(
                path, custom_cache_directory=str(process_index)
            )
            global_pytest_cache.update(cache_json)
            if res.failed:
                if is_nightly_context():
                    print(f"Suppressed Failed Nightly Sagemaker Tests")
                    print(f"{pytest_command} failed with error code: {res.return_code}\n")
                    print(f"Traceback:\n{res.stdout}")
                else:
                    raise DLCSageMakerRemoteTestFailure(
                        f"{pytest_command} failed with error code: {res.return_code}\n"
                        f"Traceback:\n{res.stdout}"
                    )
    return None


def generate_empty_report(report, test_type, case):
    """
    Generate empty junitxml report if no tests are run
    :param report: CodeBuild Report
    Returns: None
    """
    test_cases = [TestCase(test_type, case, 1, f"Skipped {test_type} on {case}", "")]
    ts = TestSuite(report, test_cases)
    with open(report, "w") as skip_file:
        TestSuite.to_file(skip_file, [ts], prettyprint=False)
