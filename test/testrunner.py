import os
import datetime
import random
import re
import sys
import logging
import traceback

from multiprocessing import Pool
import boto3
import pytest

from botocore.config import Config
from invoke import run
from invoke.context import Context

import test_utils
from test_utils import ec2 as ec2_utils
from test_utils import eks as eks_utils
from test_utils import get_dlc_images, is_pr_context, destroy_ssh_keypair, setup_sm_benchmark_tf_train_env
from test_utils import KEYS_TO_DESTROY_FILE, UBUNTU_16_BASE_DLAMI, SAGEMAKER_LOCAL_TEST_TYPE, \
    SAGEMAKER_REMOTE_TEST_TYPE, UBUNTU_HOME_DIR


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
    local_test_report = os.path.join(UBUNTU_HOME_DIR, "test", f"{tag}_local.xml")

    remote_pytest_cmd = (f"pytest {integration_path} --region {region} {docker_base_arg} "
                         f"{sm_remote_docker_base_name} --tag {tag} {aws_id_arg} {account_id} "
                         f"{instance_type_arg} {instance_type} --junitxml {test_report}")

    local_pytest_cmd = (f"python3 -m pytest {integration_path} --region {region} {docker_base_arg} "
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
    ec2_test_report_path = os.path.join(UBUNTU_HOME_DIR, "test", f"{tag}_local.xml")
    try:
        key_file = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)
        instance_id, ip_address = launch_sagemaker_local_ec2_instance(image, UBUNTU_16_BASE_DLAMI, ec2_key_name, region)
        ec2_conn = ec2_utils.ec2_connection(instance_id, key_file, region)
        run(f"tar -cz --exclude='*.pytest_cache' -f {sm_tests_tar_name} {sm_tests_path}")
        ec2_conn.put(sm_tests_tar_name, f"{UBUNTU_HOME_DIR}")
        ec2_conn.run(f"$(aws ecr get-login --no-include-email --region {region})")
        ec2_conn.run(f"tar -xzf {sm_tests_tar_name}")
        # ec2_conn.run(f"docker pull {image}")
        with ec2_conn.cd(path):
            ec2_conn.run("sudo pip3 install --upgrade -r requirements.txt ", warn=True)
            ec2_conn.run(pytest_command)
    finally:
        ec2_utils.terminate_instance(instance_id, region)
        test_utils.destroy_ssh_keypair(ec2_client, ec2_key_name)
    return True


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
            result = p.map(run_sagemaker_local_tests, images)
            if not result:
                raise Exception("Sagemaker Local tests failed")


def pull_dlc_images(images):
    """
    Pulls DLC images to CodeBuild jobs before running PyTest commands
    """
    for image in images:
        run(f"docker pull {image}", hide="out")


def setup_eks_clusters(dlc_images):
    frameworks = {"tensorflow": "tf", "pytorch": "pt", "mxnet": "mx"}
    frameworks_in_images = [framework for framework in frameworks.keys() if framework in dlc_images]
    if len(frameworks_in_images) != 1:
        raise ValueError(
            f"All images in dlc_images must be of a single framework for EKS tests.\n"
            f"Instead seeing {frameworks_in_images} frameworks."
        )
    long_name = frameworks_in_images[0]
    short_name = frameworks[long_name]
    num_nodes = 2 if is_pr_context() else 3 if long_name != "pytorch" else 4
    cluster_name = f"dlc-{short_name}-cluster-" \
                   f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-{random.randint(1, 10000)}"
    eks_utils.create_eks_cluster(cluster_name, "gpu", num_nodes, "p3.16xlarge", "pytest.pem")
    eks_utils.eks_setup(long_name, cluster_name)
    return cluster_name


def setup_sm_benchmark_env(dlc_images, test_path):
    # The plan is to have a separate if/elif-condition for each type of image
    if "tensorflow-training" in dlc_images:
        tf1_images_in_list = (re.search(r"tensorflow-training:(^ )*1(\.\d+){2}", dlc_images) is not None)
        tf2_images_in_list = (re.search(r"tensorflow-training:(^ )*2(\.\d+){2}", dlc_images) is not None)
        resources_location = os.path.join(test_path, "tensorflow", "training", "resources")
        setup_sm_benchmark_tf_train_env(resources_location, tf1_images_in_list, tf2_images_in_list)


def main():
    # Define constants
    test_type = os.getenv("TEST_TYPE")
    dlc_images = get_dlc_images()
    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]
    new_eks_cluster_name = None
    benchmark_mode = "benchmark" in test_type
    specific_test_type = re.sub("benchmark-", "", test_type) if benchmark_mode else test_type
    test_path = os.path.join("benchmark", specific_test_type) if benchmark_mode else specific_test_type

    if specific_test_type in ("sanity", "ecs", "ec2", "eks"):
        report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")

        # PyTest must be run in this directory to avoid conflicting w/ sagemaker_tests conftests
        os.chdir(os.path.join("test", "dlc_tests"))

        # Pull images for necessary tests
        if specific_test_type == "sanity":
            pull_dlc_images(all_image_list)
        if specific_test_type == "eks":
            new_eks_cluster_name = setup_eks_clusters(dlc_images)
        # Execute dlc_tests pytest command
        pytest_cmd = ["-s", "-rA", test_path, f"--junitxml={report}", "-n=auto"]
        try:
            sys.exit(pytest.main(pytest_cmd))
        finally:
            if specific_test_type == "eks":
                eks_utils.delete_eks_cluster(new_eks_cluster_name)

            # Delete dangling EC2 KeyPairs
            if specific_test_type == "ec2" and os.path.exists(KEYS_TO_DESTROY_FILE):
                with open(KEYS_TO_DESTROY_FILE) as key_destroy_file:
                    for key_file in key_destroy_file:
                        LOGGER.info(key_file)
                        ec2_client = boto3.client("ec2", config=Config(retries={'max_attempts': 10}))
                        if ".pem" in key_file:
                            _resp, keyname = destroy_ssh_keypair(ec2_client, key_file)
                            LOGGER.info(f"Deleted {keyname}")
    elif specific_test_type == "sagemaker":
        if benchmark_mode:
            report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
            os.chdir(os.path.join("test", "dlc_tests"))

            setup_sm_benchmark_env(dlc_images, test_path)
            pytest_cmd = ["-s", "-rA", test_path, f"--junitxml={report}", "-n=auto", "-o", "norecursedirs=resources"]
            sys.exit(pytest.main(pytest_cmd))
        else:
            run_sagemaker_tests(
                [image for image in standard_images_list if not ("tensorflow-inference" in image and "py2" in image)]
            )
    else:
        raise NotImplementedError(f"{test_type} test is not supported. "
                                  f"Only support ec2, ecs, eks, sagemaker and sanity currently")


if __name__ == "__main__":
    main()
