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
from test_utils import get_dlc_images, is_pr_context, destroy_ssh_keypair, setup_sm_benchmark_tf_train_env
from test_utils import KEYS_TO_DESTROY_FILE


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
        f"python3 -m pytest {integration_path} --region {region} {docker_base_arg} "
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
    print(pytest_command)

    context = Context()
    with context.cd(path):
        context.run(f"python3 -m virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            context.run(pytest_command)
            print("running test commands succeed")


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
    dlc_images = os.getenv("DLC_IMAGES")
    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    print(all_image_list)
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]

    # NOTE: runnign all_image_list now for testing purpose. Will change to standard_images_list. 
    if test_type == "sagemaker":
        run_sagemaker_tests(
            [image for image in all_image_list if not ("tensorflow-inference" in image and "py2" in image)]
        )
    else:
        raise NotImplementedError(f"{test_type} test is not supported. "
                                  f"Only support ec2, ecs, eks, sagemaker and sanity currently")


if __name__ == "__main__":
    main()

