import os
import sys

from multiprocessing import Pool

import pytest
import subprocess


def run_sagemaker_pytest_cmd(image):
    """
    Function to determine the correct pytest command to run for SageMaker tests

    :param image: ECR url
    :param instance_type: ml instance type to run on. Ex: ml.p3.16xlarge
    """

    region = os.getenv("AWS_REGION", "us-west-2")
    integration_path = os.path.join("integration", "sagemaker")

    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    docker_base_name, tag = image.split("/")[1].split(":")

    # Assign instance type
    instance_type = "ml.p3.16xlarge" if "gpu" in tag else "ml.c5.18xlarge"

    # Get path to test directory
    find_path = docker_base_name.split("-")

    # We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework = find_path[1]
    job_type = find_path[2]
    path = os.path.join("sagemaker_tests", framework, job_type)
    os.chdir(path)
    cmd = [
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


    try:
        subprocess.run(f"virtualenv {tag}", shell=True)
        subprocess.run(f"source {tag}/bin/activate && pip install -r requirements.txt && deactivate", shell=True)

    except subprocess.CalledProcessError as e:
        raise EnvironmentError("Can't activate the virtual env for running pytest commands")
    # pytest.main(cmd)

    return "Not running pytest until DLC-529 is implemented"


def run_sagemaker_tests(images):
    """
    Function to set up multiprocessing for SageMaker tests

    :param images:
    :return:
    """
    pool_number = len(images)
    with Pool(pool_number) as p:
        p.map(run_sagemaker_pytest_cmd, images)


def main():
    # Define constants
    test_type = os.getenv("TEST_TYPE")
    dlc_images = os.getenv("DLC_IMAGES")

    if test_type == "sanity":
        os.chdir("dlc_tests")
        sys.exit(pytest.main([test_type]))
    elif test_type == "sagemaker":
        run_sagemaker_tests(dlc_images.split(" "))
    else:
        raise NotImplementedError("Tests only support sagemaker and sanity currently")


if __name__ == "__main__":
    main()
