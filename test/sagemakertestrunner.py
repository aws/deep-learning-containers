import logging
import os
import re
import sys

import log_return
from invoke import run
from invoke.context import Context
from test_utils import setup_sm_benchmark_tf_train_env

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def assign_sagemaker_instance_type(image):
    """
    Assigning the instance type that the input image needs for testing

    :param image: <string> ECR URI
    :return: <string> type of instance used by the image
    """
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
        instance_type
    )



def run_sagemaker_tests(image, num_of_instances):
    """
    Run pytest in a virtual env for a particular image

    Expected to run via multiprocessing

    :param image: ECR url
    """
    pytest_command, path, tag, instance_type = generate_sagemaker_pytest_cmd(image)
    job_type = "training" if "training" in image else "inference"

    #update resource pool accordingly, then add a try-catch statement here to update the pool in case of failure
    try:
        log_return.update_pool("running", instance_type, num_of_instances, job_type)
        context = Context()
        with context.cd(path):
            context.run(f"python3 -m virtualenv {tag}")
            with context.prefix(f"source {tag}/bin/activate"):
                context.run("pip install -r requirements.txt", warn=True)
                context.run(pytest_command)
    except Exception as e:
        log_return.update_pool("runtimeError", instance_type, num_of_instances, job_type)
        raise e


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
    dlc_image = os.getenv("DLC_IMAGE")
    LOGGER.info(f"Images tested: {dlc_image}")

    num_of_instances = os.getenv("NUM_INSTANCES")
    job_type = "training" if "training" in dlc_image else "inference"

    instance_type = assign_sagemaker_instance_type(dlc_image)

    if test_type == "sagemaker":
        run_sagemaker_tests(dlc_image, num_of_instances)
    else:
        raise NotImplementedError(f"{test_type} test is not supported. Only support sagemaker currently")

    # sending log back to SQS queue
    tag = dlc_image.split("/")[-1].split(":")[-1]
    test_report = os.path.join(os.getcwd(), "test", f"{tag}.xml")
    log_return.send_log(test_report)

    # update in-progress pool
    log_return.update_pool("completed", instance_type, num_of_instances, job_type)


if __name__ == "__main__":
    main()
