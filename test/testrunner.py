import os
import sys

from multiprocessing import Pool

import pytest

from invoke.context import Context


def run_sagemaker_pytest_cmd(image):
    """
    Run pytest in a virtual env for a particular image
    Expected to run via multiprocessing
    :param image: ECR url
    """

    region = os.getenv("AWS_REGION", "us-west-2")
    integration_path = os.path.join("integration", "sagemaker")
    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    docker_base_name, tag = image.split("/")[1].split(":")

    # Assign instance type
    instance_type = "ml.p3.16xlarge" if "gpu" in tag else "ml.c4.8xlarge"

    # Get path to test directory
    find_path = docker_base_name.split("-")

    # NOTE: We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework = find_path[1]
    job_type = find_path[2]
    path = os.path.join("sagemaker_tests", framework, job_type)

    # NOTE: We are relying on tag structure to get TF major version. If tagging changes, this will break.
    if framework == "tensorflow" and job_type == "training":
        tf_major_version = tag.split("-")[-1].split(".")[0]
        path = os.path.join(
            "sagemaker_tests", framework, f"{framework}{tf_major_version}_training"
        )

    test_report = os.path.join(os.getcwd(), f"{tag}.xml")
    pytest_command = f"pytest {integration_path} --region {region} --docker-base-name " \
                        f"{docker_base_name} --tag {tag} --aws-id {account_id} --instance-type {instance_type} " \
                        f"--junitxml {test_report}"
    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            if framework != "tensorflow" and "cpu" in tag:
                context.run(pytest_command)


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
        report = os.path.join(os.getcwd(), 'sanity.xml')
        os.chdir("dlc_tests")
        sys.exit(pytest.main([test_type, f"--junitxml={report}"]))
    elif test_type == "sagemaker":
        run_sagemaker_tests(dlc_images.split(" "))
    else:
        raise NotImplementedError("Tests only support sagemaker and sanity currently")


if __name__ == "__main__":
    main()