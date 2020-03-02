import argparse
import os

import pytest


def get_args():
    """
    Manage arguments to this script when called directly
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        nargs='+',
        required=True,
        help="DLC_IMAGES env variable -- list of space separated ECR urls"
    )
    parser.add_argument(
        "--test-type",
        required=True,
        choices=['sagemaker', 'sanity', 'ecs', 'ec2', 'eks'],
        help="Type of test to run -- will indicate which CodeBuild job to run"
    )
    return parser.parse_args()


def run_sagemaker_tests(images):
    """
    Function to determine the correct pytest command to run for SageMaker tests

    :param images: list of ECR urls
    """
    region = os.getenv('AWS_REGION', 'us-west-2')
    integration_path = os.path.join("integration", "sagemaker")
    for image in images:
        sm_test_args, path = parse_sagemaker_image_tags(image)
        cmd = [f"--rootdir={path}", integration_path, "--region", region] + sm_test_args
        pytest.main(cmd)


def parse_sagemaker_image_tags(image):
    """
    Function to parse the SageMaker image tags

    :param image: ECR url for image
    :return: arguments to pass to pytest.main
    """
    # args = {}
    account_id = image.split('.')[0]
    docker_base_name, tag = image.split('/')[1].split(':')

    # Get path to test directory
    find_path = docker_base_name.split('-')
    path = os.path.join('sagemaker', find_path[1], find_path[2])
    return ["--docker-base-name", docker_base_name, "--tag", tag, "--aws-id", account_id], path


def main():
    args = get_args()
    test_type = args.test_type
    dlc_images = args.images
    if test_type != "sagemaker":
        pytest.main([test_type, "--images", dlc_images])
    else:
        run_sagemaker_tests(dlc_images)


if __name__ == "__main__":
    main()
