import datetime
import os
import traceback
import random
import re

import boto3
import pytest
from invoke.context import Context
from invoke import run

from test_utils import ec2 as ec2_utils
from test_utils import destroy_ssh_keypair, generate_ssh_keypair
from test_utils import UBUNTU_16_BASE_DLAMI, SAGEMAKER_LOCAL_TEST_TYPE, \
    SAGEMAKER_REMOTE_TEST_TYPE, UBUNTU_HOME_DIR, DEFAULT_REGION

def assign_sagemaker_remote_job_instance_type(image):
    if "tensorflow" in image:
        return "ml.p3.8xlarge" if "gpu" in image else "ml.c4.4xlarge"
    else:
        return "ml.p2.8xlarge" if "gpu" in image else "ml.c4.8xlarge"


def assign_sagemaker_local_job_instance_type(image):
    if "training" in image:
        return "p3.8xlarge" if "gpu" in image else "c5.18xlarge"
    else:
        return "p2.xlarge" if "gpu" in image else "c5.18xlarge"


def launch_sagemaker_local_ec2_instance(image, ami_id, ec2_key_name, region):
    instance_type = assign_sagemaker_local_job_instance_type(image)
    instance_name = image.split(":")[-1]
    instance = ec2_utils.launch_instance(
        ami_id,
        region=region,
        ec2_key_name=ec2_key_name,
        instance_type=instance_type,
        user_data=None,
        iam_instance_profile_name=ec2_utils.EC2_INSTANCE_ROLE_NAME,
        instance_name=f"{instance_name}",
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
    :return: <tuple> pytest command to be run, path where it should be executed, image tag
    """
    reruns = 4
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    integration_path = os.path.join("integration", sagemaker_test_type)
    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    sm_remote_docker_base_name, tag = image.split("/")[1].split(":")
    sm_local_docker_base_name = image.split(":")[0]

    # Assign instance type
    instance_type = assign_sagemaker_remote_job_instance_type(image)

    # Get path to test directory
    find_path = sm_remote_docker_base_name.split("-")

    # NOTE: We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework = find_path[1]
    job_type = find_path[2]
    path = os.path.join("test", "sagemaker_tests", framework, job_type)
    aws_id_arg = "--aws-id"
    docker_base_arg = "--docker-base-name"
    instance_type_arg = "--instance-type"
    framework_version = re.search(r"\d+(\.\d+){2}", tag).group()
    framework_major_version = framework_version.split(".")[0]
    processor = "gpu" if "gpu" in image else "cpu"
    py_version = re.search(r"py(\d)+", tag).group()

    # Conditions for modifying tensorflow SageMaker pytest commands
    if framework == "tensorflow" and sagemaker_test_type == SAGEMAKER_REMOTE_TEST_TYPE:
        if job_type == "training":
            aws_id_arg = "--account-id"
        else:
            aws_id_arg = "--registry"
            docker_base_arg = "--repo"
            integration_path = os.path.join(integration_path, "test_tfs.py")
            instance_type_arg = "--instance-types"

    test_report = os.path.join(os.getcwd(), "test", f"{tag}.xml")
    local_test_report = os.path.join(UBUNTU_HOME_DIR, "test", f"{job_type}_{tag}_sm_local.xml")
    is_py3 = " python3 -m "

    remote_pytest_cmd = (f"pytest {integration_path} --region {region} {docker_base_arg} "
                         f"{sm_remote_docker_base_name} --tag {tag} {aws_id_arg} {account_id} "
                         f"{instance_type_arg} {instance_type} --junitxml {test_report}")

    local_pytest_cmd = (f"{is_py3} pytest {integration_path} --region {region} {docker_base_arg} "
                        f"{sm_local_docker_base_name} --tag {tag} --framework-version {framework_version} "
                        f"--processor {processor} --junitxml {local_test_report}")

    if framework == "tensorflow" and job_type != "inference":
        local_pytest_cmd = f"{local_pytest_cmd} --py-version {py_version[2]}"
    if framework == "tensorflow" and job_type == "training":
        path = os.path.join(os.path.dirname(path), f"{framework}{framework_major_version}_training")


    return (
        remote_pytest_cmd if sagemaker_test_type == SAGEMAKER_REMOTE_TEST_TYPE else local_pytest_cmd,
        path,
        tag,
        job_type
    )


def run_sagemaker_local_tests(ec2_client, image, region):
    """
    Run the sagemaker local tests in ec2 instance for the image
    :param image: ECR url
    :return: None
    """
    pytest_command, path, tag, job_type = generate_sagemaker_pytest_cmd(image, SAGEMAKER_LOCAL_TEST_TYPE)
    framework = image.split("/")[1].split(":")[0].split("-")[1]
    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    ec2_key_name = f"{tag}_sagemaker_{random.randint(1,1000)}"
    sm_tests_tar_name = "sagemaker_tests.tar.gz"
    ec2_test_report_path = os.path.join(UBUNTU_HOME_DIR, "test", f"{job_type}_{tag}_sm_local.xml")
    try:
        key_file = generate_ssh_keypair(ec2_client, ec2_key_name)
        print(f"Launching new Instance for image: {image}")
        instance_id, ip_address = launch_sagemaker_local_ec2_instance(image, UBUNTU_16_BASE_DLAMI, ec2_key_name, region)
        ec2_conn = ec2_utils.ec2_connection(instance_id, key_file, region)
        run(f"tar -cz --exclude='*.pytest_cache' -f {sm_tests_tar_name} {sm_tests_path}")
        ec2_conn.put(sm_tests_tar_name, f"{UBUNTU_HOME_DIR}")
        ec2_conn.run(f"$(aws ecr get-login --no-include-email --region {region})")
        ec2_conn.run(f"tar -xzf {sm_tests_tar_name}")
        is_py3 = " python3 -m"
        with ec2_conn.cd(path):
            ec2_conn.run(f"sudo {is_py3} pip install -U pytest pytest-xdist boto3 requests pytest-rerunfailures")
            # To avoid the dpkg lock for apt remove
            ec2_conn.run("sleep 2m")
            ec2_conn.run("sudo apt-get remove python3-scipy python3-yaml -y")
            ec2_conn.run(f"sudo {is_py3} pip install -r requirements.txt ", warn=True)
            ec2_conn.run(pytest_command, timeout=2100)
            print(f"Downloading Test reports for image: {image}")
            ec2_conn.get(ec2_test_report_path, os.path.join("test", f"{job_type}_{tag}_sm_local.xml"))
            print(f"Test reports downloaded for image: {image}")
    finally:
        print(f"Terminating Instances for image: {image}")
        ec2_utils.terminate_instance(instance_id, region)
        print(f"Destroying ssh Key_pair for image: {image}")
        destroy_ssh_keypair(ec2_client, ec2_key_name)


def run_sagemaker_remote_tests(image):
    """
    Run pytest in a virtual env for a particular image

    Expected to run via multiprocessing

    :param image: ECR url
    """
    pytest_command, path, tag, job_type = generate_sagemaker_pytest_cmd(image, SAGEMAKER_REMOTE_TEST_TYPE)
    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            context.run(pytest_command)

