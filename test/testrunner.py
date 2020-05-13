import os
import datetime
import random
import re
import sys
import logging

from multiprocessing import Pool
import boto3
import pytest

from botocore.config import Config
from invoke import run
from invoke.context import Context

import test_utils
from test_utils import ec2 as ec2_utils
from test_utils import eks as eks_utils
from test_utils import get_dlc_images, is_pr_context, destroy_ssh_keypair, KEYS_TO_DESTROY_FILE, \
    SAGEMAKER_AMI_ID, SAGEMAKER_LOCAL_TEST_TYPE, SAGEMAKER_REMOTE_TEST_TYPE, AML_HOME_DIR

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


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
    region = os.getenv("AWS_REGION", "us-west-2")
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
    local_test_report = os.path.join(AML_HOME_DIR, "test", f"{tag}_local.xml")

    remote_pytest_cmd = (f"pytest --reruns {reruns} {integration_path} --region {region} {docker_base_arg} "
                         f"{sm_remote_docker_base_name} --tag {tag} {aws_id_arg} {account_id} "
                         f"{instance_type_arg} {instance_type} --junitxml {test_report}")

    local_pytest_cmd = (f"python3 -m pytest --reruns {reruns} {integration_path} --region {region} {docker_base_arg} "
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
    )


def run_sagemaker_local_tests(image):
    """
    Run the sagemaker local tests in ec2 instance for the image
    :param image: ECR url
    :return: None
    """
    region = os.getenv("AWS_REGION", "us-west-2")
    pytest_command, path, tag = generate_sagemaker_pytest_cmd(image, SAGEMAKER_LOCAL_TEST_TYPE)
    framework = image.split("/")[1].split(":")[0].split("-")[1]
    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    ec2_key_name = f"{tag}_sagemaker_{random.randint(1,1000)}"
    ec2_client = ec2_utils.ec2_client(region)
    sm_tests_path = os.path.join("test", "sagemaker_tests", framework)
    sm_tests_tar_name = "sagemaker_tests.tar.gz"
    ec2_test_report_path = os.path.join(AML_HOME_DIR, "test", f"{tag}_local.xml")
    try:
        key_file = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
        instance_id, ip_address = launch_sagemaker_local_ec2_instance(image, SAGEMAKER_AMI_ID, ec2_key_name, region)
        ec2_conn = ec2_utils.ec2_connection(instance_id, key_file, region)
        run(f"tar -czf {sm_tests_tar_name} {sm_tests_path}")
        ec2_conn.put(sm_tests_tar_name, f"{AML_HOME_DIR}")
        ec2_conn.run(f"$(aws ecr get-login --no-include-email --region {region})")
        ec2_conn.run(f"docker pull {image}")
        ec2_conn.run(f"tar -xvf {sm_tests_tar_name}")
        with ec2_conn.cd(path):
            ec2_conn.run("sudo pip3 install -r requirements.txt ", warn=True)
            ec2_conn.run(pytest_command)
            ec2_conn.get(ec2_test_report_path, f"test/{tag}_local.xml")
    finally:
        ec2_utils.terminate_instance(instance_id, region)
        test_utils.destroy_ssh_keypair(ec2_client, ec2_key_name)


def run_sagemaker_remote_tests(image):
    """
    Run pytest in a virtual env for a particular image

    Expected to run via multiprocessing

    :param image: ECR url
    """
    pytest_command, path, tag = generate_sagemaker_pytest_cmd(image, SAGEMAKER_REMOTE_TEST_TYPE)
    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            context.run(pytest_command)


def run_sagemaker_tests(images):
    """
    Function to set up multiprocessing for SageMaker tests

    :param images: <list> List of all images to be used in SageMaker tests
    """
    if not images:
        return
    pool_number = len(images)
    with Pool(pool_number) as p:
        # p.map(run_sagemaker_remote_tests, images)
        if is_pr_context():
            p.map(run_sagemaker_local_tests, images)


def pull_dlc_images(images):
    """
    Pulls DLC images to CodeBuild jobs before running PyTest commands
    """
    for image in images:
        run(f"docker pull {image}", hide="out")


def setup_eks_clusters(dlc_images):
    terminable_clusters = []
    frameworks = {"tensorflow": "tf", "pytorch": "pt", "mxnet": "mx"}
    for long_name, short_name in frameworks.items():
        if long_name in dlc_images:
            cluster_name = None
            if not is_pr_context():
                num_nodes = 3 if long_name != "pytorch" else 4
                cluster_name = f"dlc-{short_name}-cluster-" \
                               f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-{random.randint(1, 10000)}"
                eks_utils.create_eks_cluster(cluster_name, "gpu", num_nodes, "p3.16xlarge", "pytest.pem")
                terminable_clusters.append(cluster_name)
            eks_utils.eks_setup(long_name, cluster_name)
    return terminable_clusters


def main():
    # Define constants
    test_type = os.getenv("TEST_TYPE")
    dlc_images = get_dlc_images()
    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]
    eks_terminable_clusters = []

    if test_type in ("sanity", "ecs", "ec2", "eks"):
        report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")

        # PyTest must be run in this directory to avoid conflicting w/ sagemaker_tests conftests
        os.chdir(os.path.join("test", "dlc_tests"))

        # Pull images for necessary tests
        if test_type == "sanity":
            pull_dlc_images(all_image_list)
        if test_type == "eks":
            eks_terminable_clusters = setup_eks_clusters(dlc_images)
        # Execute dlc_tests pytest command
        pytest_cmd = ["-s", "-rA", test_type, f"--junitxml={report}", "-n=auto"]
        try:
            sys.exit(pytest.main(pytest_cmd))
        finally:
            if test_type == "eks" and eks_terminable_clusters:
                for cluster in eks_terminable_clusters:
                    eks_utils.delete_eks_cluster(cluster)

            # Delete dangling EC2 KeyPairs
            if test_type == "ec2" and os.path.exists(KEYS_TO_DESTROY_FILE):
                with open(KEYS_TO_DESTROY_FILE) as key_destroy_file:
                    for key_file in key_destroy_file:
                        LOGGER.info(key_file)
                        ec2_client = boto3.client("ec2", config=Config(retries={'max_attempts': 10}))
                        if ".pem" in key_file:
                            destroy_ssh_keypair(ec2_client, key_file)
    elif test_type == "sagemaker":
        run_sagemaker_tests(
            [image for image in standard_images_list if not ("tensorflow-inference" in image and "py2" in image)]
        )
    else:
        raise NotImplementedError("Tests only support sagemaker and sanity currently")


if __name__ == "__main__":
    main()
