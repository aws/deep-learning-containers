import argparse
import os
import sys

from multiprocessing import Pool

import pytest


def run_sagemaker_pytest_cmd(image, instance_type):
    """
    Function to determine the correct pytest command to run for SageMaker tests

    :param image: ECR url
    :param instance_type: ml instance type to run on. Ex: ml.p3.16xlarge
    """
    region = os.getenv("AWS_REGION", "us-west-2")
    integration_path = os.path.join("integration", "sagemaker")

    account_id = image.split(".")[0]
    docker_base_name, tag = image.split("/")[1].split(":")
    # Get path to test directory
    find_path = docker_base_name.split("-")
    path = os.path.join("sagemaker", find_path[1], find_path[2])

    cmd = [
        f"--rootdir={path}",
        integration_path,
        "--region",
        region,
        "--docker-base-name",
        docker_base_name,
        "--tag",
        tag,
        "--aws-id",
        account_id,
        "--instance-type",
        instance_type,
    ]
    # pytest.main(cmd)
    return "Not running pytest until DLC-529 is implemented"


def run_sagemaker_tests(images):
    """
    Function to set up multiprocessing for SageMaker tests

    :param images:
    :return:
    """
    sagemaker_instance_types = ("ml.p3.16xlarge", "ml.c5.18xlarge")
    sagemaker_params = []
    for image in images:
        sagemaker_params.append((image, sagemaker_instance_types[0]))
        sagemaker_params.append((image, sagemaker_instance_types[1]))

    pool_number = len(sagemaker_params)
    with Pool(pool_number) as p:
        p.starmap(run_sagemaker_pytest_cmd, sagemaker_params)


def main():
    # Define constants
    test_type = os.getenv('TEST_TYPE')
    dlc_images = os.getenv("DLC_IMAGES")

    if test_type == "sanity":
        sys.exit(pytest.main([test_type, f"--images={dlc_images}"]))
    elif test_type == "sagemaker":
        run_sagemaker_tests(dlc_images)
    else:
        raise NotImplementedError("Tests only support sagemaker and sanity currently")


if __name__ == "__main__":
    main()
