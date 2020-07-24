import os
import random
import sys
import logging
import re
import traceback

from multiprocessing import Pool
import boto3
import pytest

from botocore.config import Config
from invoke import run

from test_utils import eks as eks_utils
from test_utils import sagemaker as sm_utils
from test_utils import (get_dlc_images,
                        is_pr_context,
                        destroy_ssh_keypair,
                        setup_sm_benchmark_tf_train_env,
                        get_framework_and_version_from_tag)
from test_utils import KEYS_TO_DESTROY_FILE, DEFAULT_REGION

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def run_sagemaker_local_tests(images):
    """
    Function to run the SageMaker Local tests
    :param images: <list> List of all images to be used in SageMaker tests
    """
    if not images:
        return
    # Run sagemaker Local tests
    framework, _ = get_framework_and_version_from_tag(images[0])
    sm_tests_path = os.path.join("test", "sagemaker_tests", framework)
    sm_tests_tar_name = "sagemaker_tests.tar.gz"
    run(f"tar -cz --exclude='*.pytest_cache' --exclude='__pycache__' -f {sm_tests_tar_name} {sm_tests_path}")
    ec2_client = boto3.client("ec2", config=Config(retries={'max_attempts': 10}), region_name=DEFAULT_REGION)
    for image in images:
        sm_utils.execute_local_tests(image, ec2_client)


def run_sagemaker_remote_tests(images):
    """
    Function to set up multiprocessing for SageMaker tests
    :param images: <list> List of all images to be used in SageMaker tests
    """
    if not images:
        return
    pool_number = len(images)
    with Pool(pool_number) as p:
        p.map(sm_utils.execute_sagemaker_remote_tests, images)


def pull_dlc_images(images):
    """
    Pulls DLC images to CodeBuild jobs before running PyTest commands
    """
    for image in images:
        run(f"docker pull {image}", hide="out")


def setup_eks_cluster(framework_name):
    frameworks = {"tensorflow": "tf", "pytorch": "pt", "mxnet": "mx"}
    long_name = framework_name
    short_name = frameworks[long_name]
    codebuild_version = os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')[0:7]
    num_nodes = 1 if is_pr_context() else 3 if long_name != "pytorch" else 4
    cluster_name = f"dlc-{short_name}-cluster-" \
                   f"{codebuild_version}-{random.randint(1, 10000)}"
    try:
        eks_utils.eks_setup()
        eks_utils.create_eks_cluster(cluster_name, "gpu", num_nodes, "p3.16xlarge", "pytest.pem")
    except Exception:
        eks_utils.delete_eks_cluster(cluster_name)
        raise
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
    eks_cluster_name = None
    benchmark_mode = "benchmark" in test_type
    specific_test_type = re.sub("benchmark-", "", test_type) if benchmark_mode else test_type
    test_path = os.path.join("benchmark", specific_test_type) if benchmark_mode else specific_test_type

    if specific_test_type in ("sanity", "ecs", "ec2", "eks", "canary"):
        report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
        # The following two report files will only be used by EKS tests, as eks_train.xml and eks_infer.xml.
        # This is to sequence the tests and prevent one set of tests from waiting too long to be scheduled.
        report_train = os.path.join(os.getcwd(), "test", f"{test_type}_train.xml")
        report_infer = os.path.join(os.getcwd(), "test", f"{test_type}_infer.xml")
        report_multinode_train = os.path.join(os.getcwd(), "test", f"eks_multinode_train.xml")

        # PyTest must be run in this directory to avoid conflicting w/ sagemaker_tests conftests
        os.chdir(os.path.join("test", "dlc_tests"))

        # Pull images for necessary tests
        if specific_test_type == "sanity":
            pull_dlc_images(all_image_list)
        if specific_test_type == "eks":
            frameworks_in_images = [framework for framework in ("mxnet", "pytorch", "tensorflow")
                                    if framework in dlc_images]
            if len(frameworks_in_images) != 1:
                raise ValueError(
                    f"All images in dlc_images must be of a single framework for EKS tests.\n"
                    f"Instead seeing {frameworks_in_images} frameworks."
                )
            framework = frameworks_in_images[0]
            eks_cluster_name = setup_eks_cluster(framework)

            # setup kubeflow
            eks_utils.setup_kubeflow(eks_cluster_name)

            # Change 1: Split training and inference, and run one after the other, to prevent scheduling issues
            # Set -n=4, instead of -n=auto, because initiating too many pods simultaneously has been resulting in
            # pods timing-out while they were in the Pending state. Scheduling 4 tests (and hence, 4 pods) at once
            # seems to be an optimal configuration.
            # Change 2: Separate multi-node EKS tests from single-node tests in execution to prevent resource contention
            pytest_cmds = [
                ["-s", "-rA", os.path.join(test_path, framework, "training"), f"--junitxml={report_train}", "-n=4",
                 "-m", "not multinode"],
                ["-s", "-rA", os.path.join(test_path, framework, "inference"), f"--junitxml={report_infer}", "-n=4"],
                ["-s", "-rA", test_path, f"--junitxml={report_multinode_train}", "--multinode"],
            ]
        else:
            # Execute dlc_tests pytest command
            pytest_cmds = [["-s", "-rA", test_path, f"--junitxml={report}", "-n=auto"]]
        # Execute separate cmd for canaries
        if specific_test_type == "canary":
            pytest_cmds = [["-s", "-rA", f"--junitxml={report}", "-n=auto", "--canary", "--ignore=container_tests/"]]
        try:
            # Note:- Running multiple pytest_cmds in a sequence will result in the execution log having two
            #        separate pytest reports, both of which must be examined in case of a manual review of results.
            cmd_exit_statuses = [pytest.main(pytest_cmd) for pytest_cmd in pytest_cmds]
            sys.exit(0) if all([status == 0 for status in cmd_exit_statuses]) else sys.exit(1)
        finally:
            if specific_test_type == "eks" and eks_cluster_name:
                eks_utils.delete_eks_cluster(eks_cluster_name)

            # Delete dangling EC2 KeyPairs
            if os.path.exists(KEYS_TO_DESTROY_FILE):
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
            run_sagemaker_remote_tests(
                [image for image in standard_images_list if not ("tensorflow-inference" in image and "py2" in image)]
            )
    elif specific_test_type == "sagemaker-local":
        run_sagemaker_local_tests(
            [image for image in standard_images_list if not ("tensorflow-inference" in image and "py2" in image)]
        )
    else:
        raise NotImplementedError(f"{test_type} test is not supported. "
                                  f"Only support ec2, ecs, eks, sagemaker and sanity currently")


if __name__ == "__main__":
    main()
