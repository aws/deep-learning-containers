import os
import sys

from multiprocessing import Pool

import pytest
import invoke

from invoke.context import Context

class SageMakerPytestError(Exception):
    """
    Custom exception for SM pytest erros
    """
    pass


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
    if framework == "tensorflow" and job_type == "training":

        # This code fetches the tag from the ecr repo with the framework version at the end.
        # NOTE: If tagging method changes, this will break
        tf_major_version = tag.split("-")[-1].split('.')[0]
        path = os.path.join("sagemaker_tests", framework, f"{framework}{tf_major_version}_training")
        print("path", path)
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
    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)

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
