import os
import random
import sys
import logging
import re

from multiprocessing import Pool

import boto3
import pytest

from botocore.config import Config
from invoke import run
from invoke.context import Context

from test_utils import eks as eks_utils
from test_utils import get_dlc_images, is_pr_context, destroy_ssh_keypair, KEYS_TO_DESTROY_FILE


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def assign_sagemaker_instance_type(image):
    if "tensorflow" in image:
        return "ml.p3.8xlarge" if "gpu" in image else "ml.c4.4xlarge"
    else:
        return "ml.p2.8xlarge" if "gpu" in image else "ml.c4.8xlarge"


def generate_sagemaker_pytest_cmd(image):
    """
    Parses the image ECR url and returns appropriate pytest command

    :param image: ECR url of image
    :return: <tuple> pytest command to be run, path where it should be executed, image tag
    """
    reruns = 4
    region = os.getenv("AWS_REGION", "us-west-2")
    integration_path = os.path.join("integration", "sagemaker")
    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    docker_base_name, tag = image.split("/")[1].split(":")

    # Assign instance type
    instance_type = assign_sagemaker_instance_type(image)

    # Get path to test directory
    find_path = docker_base_name.split("-")

    # NOTE: We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework = find_path[1]
    job_type = find_path[2]
    path = os.path.join("test", "sagemaker_tests", framework, job_type)
    aws_id_arg = "--aws-id"
    docker_base_arg = "--docker-base-name"
    instance_type_arg = "--instance-type"

    # Conditions for modifying tensorflow SageMaker pytest commands
    if framework == "tensorflow":
        if job_type == "training":
            aws_id_arg = "--account-id"

            # NOTE: We rely on Framework Version being in "major.minor.patch" format
            tf_framework_version = re.search(r"\d+(\.\d+){2}", tag).group()
            tf_major_version = tf_framework_version.split(".")[0]
            path = os.path.join(os.path.dirname(path), f"{framework}{tf_major_version}_training")
        else:
            aws_id_arg = "--registry"
            docker_base_arg = "--repo"
            integration_path = os.path.join(integration_path, "test_tfs.py")
            instance_type_arg = "--instance-types"

    test_report = os.path.join(os.getcwd(), "test", f"{tag}.xml")
    return (
        f"pytest --reruns {reruns} {integration_path} --region {region} {docker_base_arg} "
        f"{docker_base_name} --tag {tag} {aws_id_arg} {account_id} {instance_type_arg} {instance_type} "
        f"--junitxml {test_report}",
        path,
        tag,
    )


def run_sagemaker_pytest_cmd(image):
    """
    Run pytest in a virtual env for a particular image

    Expected to run via multiprocessing

    :param image: ECR url
    """
    pytest_command, path, tag = generate_sagemaker_pytest_cmd(image)

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
        p.map(run_sagemaker_pytest_cmd, images)


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
