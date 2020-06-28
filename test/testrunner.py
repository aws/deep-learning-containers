import os
import random
import sys
import logging
import re
import traceback

from multiprocessing import Pool
import concurrent.futures
import boto3
import pytest

from botocore.config import Config
from invoke import run

from test_utils import eks as eks_utils
from test_utils import sagemaker as sm_utils
from test_utils import get_dlc_images, is_pr_context, destroy_ssh_keypair, setup_sm_benchmark_tf_train_env
from test_utils import KEYS_TO_DESTROY_FILE, DEFAULT_REGION

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def run_sagemaker_tests(images):
    """
    Function to set up multiprocessing for SageMaker tests
    :param images: <list> List of all images to be used in SageMaker tests
    """
    if not images:
        return
    # This is to ensure that threads don't lock
    pool_number = (len(images)*2)
    with Pool(pool_number) as p:
        # p.map(sm_utils.run_sagemaker_remote_tests, images)
        # Run sagemaker Local tests
        p.map(sm_utils.run_sagemaker_local_tests, images)


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

    if specific_test_type in ("sanity", "ecs", "ec2", "eks", "canary"):
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

        # Execute separate cmd for canaries
        if specific_test_type == "canary":
            pytest_cmd = ["-s", "-rA", f"--junitxml={report}", "-n=auto", "--canary", "--ignore=container_tests/"]
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
