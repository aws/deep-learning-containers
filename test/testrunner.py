import json
import os
import random
import sys
import logging
import re

from junit_xml import TestSuite, TestCase
from multiprocessing import Pool, Manager
from datetime import datetime

import boto3
import pytest

from botocore.config import Config
from invoke import run
from invoke.context import Context
from test_utils import eks as eks_utils
from test_utils import sagemaker as sm_utils
from test_utils import metrics as metrics_utils
from test_utils import (
    get_dlc_images,
    is_pr_context,
    is_benchmark_dev_context,
    is_rc_test_context,
    is_efa_dedicated,
    is_ec2_image,
    destroy_ssh_keypair,
    setup_sm_benchmark_tf_train_env,
    setup_sm_benchmark_mx_train_env,
    setup_sm_benchmark_hf_infer_env,
    get_framework_and_version_from_tag,
    get_build_context,
    is_nightly_context,
    get_ecr_repo_name,
    generate_unique_dlc_name,
)
from test_utils import KEYS_TO_DESTROY_FILE, DEFAULT_REGION
from test_utils.pytest_cache import PytestCache

from src.codebuild_environment import get_codebuild_project_name

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
pytest_cache_util = PytestCache(
    boto3.client("s3"), boto3.client("sts").get_caller_identity()["Account"]
)


def run_sagemaker_local_tests(images, pytest_cache_params):
    """
    Function to run the SageMaker Local tests
    :param images: <list> List of all images to be used in SageMaker tests
    :param pytest_cache_params: <dict> dictionary with data required for pytest cache handler
    """
    if not images:
        return
    # Run sagemaker Local tests
    framework, _ = get_framework_and_version_from_tag(images[0])
    framework = framework.replace("_trcomp", "")
    sm_tests_path = (
        os.path.join("test", "sagemaker_tests", framework)
        if "huggingface" not in framework
        else os.path.join("test", "sagemaker_tests", "huggingface*")
    )
    sm_tests_tar_name = "sagemaker_tests.tar.gz"
    run(
        f"tar -cz --exclude='*.pytest_cache' --exclude='__pycache__' -f {sm_tests_tar_name} {sm_tests_path}"
    )

    pool_number = len(images)
    with Pool(pool_number) as p:
        test_results = p.starmap(
            sm_utils.execute_local_tests, [[image, pytest_cache_params] for image in images]
        )
    if not all(test_results):
        failed_images = [images[index] for index, result in enumerate(test_results) if not result]
        raise RuntimeError(
            f"SageMaker Local tests failed on the following DLCs:\n"
            f"{json.dumps(failed_images, indent=4)}"
        )


def run_sagemaker_test_in_executor(image, num_of_instances, instance_type):
    """
    Run pytest in a virtual env for a particular image

    Expected to run under multi-threading

    :param num_of_instances: <int> number of instances the image test requires
    :param instance_type: type of sagemaker instance the test needs
    :param image: ECR url
    :return:
    """
    import log_return

    LOGGER.info("Started running SageMaker test.....")
    pytest_command, path, tag, job_type = sm_utils.generate_sagemaker_pytest_cmd(image, "sagemaker")

    # update resource pool accordingly, then add a try-catch statement here to update the pool in case of failure
    try:
        log_return.update_pool("running", instance_type, num_of_instances, job_type)
        context = Context()
        with context.cd(path):
            context.run(f"python3 -m virtualenv {tag}")
            with context.prefix(f"source {tag}/bin/activate"):
                context.run("pip install -r requirements.txt", warn=True)
                context.run(pytest_command)
    except Exception as e:
        LOGGER.error(e)
        return False

    return True


def print_log_stream(logs):
    """
    print the log stream from Job Executor
    :param logs: <dict> the returned dict from JobRequester.receive_logs
    """
    LOGGER.info("Log stream from Job Executor.....")
    print(logs["LOG_STREAM"])
    LOGGER.info("Print log stream complete.")


def send_scheduler_requests(requester, image):
    """
    Send a PR test request through the requester, and wait for the response.
    If test completed or encountered runtime error, create local XML reports.
    Otherwise the test failed, print the failure reason.

    :param requester: JobRequester object
    :param image: <string> ECR URI
    """
    # Note: 3 is the max number of instances required for any tests. Here we schedule tests conservatively.
    identifier = requester.send_request(image, "PR", 3)
    image_tag = image.split(":")[-1]
    report_path = os.path.join(os.getcwd(), "test", f"{image_tag}.xml")
    while True:
        query_status_response = requester.query_status(identifier)
        test_status = query_status_response["status"]
        if test_status == "completed":
            LOGGER.info(f"Test for image {image} completed.")
            logs_response = requester.receive_logs(identifier)
            LOGGER.info(
                f"Receive logs success for ticket {identifier.ticket_name}, report path: {report_path}"
            )
            print_log_stream(logs_response)
            metrics_utils.send_test_result_metrics(0)
            with open(report_path, "w") as xml_report:
                xml_report.write(logs_response["XML_REPORT"])
            break

        elif test_status == "runtimeError":
            logs_response = requester.receive_logs(identifier)
            with open(report_path, "w") as xml_report:
                xml_report.write(logs_response["XML_REPORT"])
            print_log_stream(logs_response)
            metrics_utils.send_test_result_metrics(1)
            raise Exception(f"Test for image {image} ran into runtime error.")
            break

        elif test_status == "failed":
            metrics_utils.send_test_result_metrics(1)
            raise Exception(
                f"Scheduling failed for image {image}. Reason: {query_status_response['reason']}"
            )
            break


def run_sagemaker_remote_tests(images, pytest_cache_params):
    """
    Function to set up multiprocessing for SageMaker tests
    :param images: <list> List of all images to be used in SageMaker tests
    """
    use_scheduler = os.getenv("USE_SCHEDULER", "False").lower() == "true"
    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"

    if executor_mode:
        LOGGER.info("entered executor mode.")
        import log_return

        num_of_instances = os.getenv("NUM_INSTANCES")
        image = images[0]
        job_type = "training" if "training" in image else "inference"
        instance_type = sm_utils.assign_sagemaker_remote_job_instance_type(image)
        test_succeeded = run_sagemaker_test_in_executor(image, num_of_instances, instance_type)

        tag = image.split("/")[-1].split(":")[-1]
        test_report = os.path.join(os.getcwd(), "test", f"{tag}.xml")

        # update in-progress pool, send the xml reports
        if test_succeeded:
            log_return.update_pool(
                "completed", instance_type, num_of_instances, job_type, test_report
            )
        else:
            log_return.update_pool(
                "runtimeError", instance_type, num_of_instances, job_type, test_report
            )
        return

    elif use_scheduler:
        LOGGER.info("entered scheduler mode.")
        import concurrent.futures
        from job_requester import JobRequester

        job_requester = JobRequester()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(images)) as executor:
            futures = [
                executor.submit(send_scheduler_requests, job_requester, image) for image in images
            ]
            for future in futures:
                future.result()
    else:
        if not images:
            return
        pool_number = len(images)
        # Using Manager().dict() since it's a thread safe dictionary
        global_pytest_cache = Manager().dict()
        try:
            with Pool(pool_number) as p:
                p.starmap(
                    sm_utils.execute_sagemaker_remote_tests,
                    [
                        [i, images[i], global_pytest_cache, pytest_cache_params]
                        for i in range(pool_number)
                    ],
                )
        finally:
            pytest_cache_util.convert_cache_json_and_upload_to_s3(
                global_pytest_cache, **pytest_cache_params
            )


def pull_dlc_images(images):
    """
    Pulls DLC images to CodeBuild jobs before running PyTest commands
    """
    for image in images:
        run(f"docker pull {image}", hide="out")


def setup_sm_benchmark_env(dlc_images, test_path):
    # The plan is to have a separate if/elif-condition for each type of image
    if re.search(r"huggingface-(tensorflow|pytorch|mxnet)-inference", dlc_images):
        resources_location = os.path.join(test_path, "huggingface", "inference", "resources")
        setup_sm_benchmark_hf_infer_env(resources_location)
    elif "tensorflow-training" in dlc_images:
        tf1_images_in_list = (
            re.search(r"tensorflow-training:(^ )*1(\.\d+){2}", dlc_images) is not None
        )
        tf2_images_in_list = (
            re.search(r"tensorflow-training:(^ )*2(\.\d+){2}", dlc_images) is not None
        )
        resources_location = os.path.join(test_path, "tensorflow", "training", "resources")
        setup_sm_benchmark_tf_train_env(resources_location, tf1_images_in_list, tf2_images_in_list)
    elif "mxnet-training" in dlc_images:
        resources_location = os.path.join(test_path, "mxnet", "training", "resources")
        setup_sm_benchmark_mx_train_env(resources_location)


def delete_key_pairs(keyfile):
    """
    Function to delete key pairs from a file in mainline context

    :param keyfile: file with all of the keys to delete
    """
    with open(keyfile) as key_destroy_file:
        for key_file in key_destroy_file:
            LOGGER.info(key_file)
            ec2_client = boto3.client("ec2", config=Config(retries={"max_attempts": 10}))
            if ".pem" in key_file:
                _resp, keyname = destroy_ssh_keypair(ec2_client, key_file)
                LOGGER.info(f"Deleted {keyname}")


def build_bai_docker_container():
    """
    Builds docker container with necessary script requirements (bash 5.0+,conda)
    """
    # Assuming we are in dlc_tests directory
    docker_dir = os.path.join("benchmark", "bai", "docker")
    ctx = Context()
    with ctx.cd(docker_dir):
        ctx.run("docker build -t bai_env_container -f Dockerfile .")


def main():
    # Define constants
    start_time = datetime.now()
    test_type = os.getenv("TEST_TYPE")

    efa_dedicated = is_efa_dedicated()
    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"
    dlc_images = os.getenv("DLC_IMAGE") if executor_mode else get_dlc_images()
    # Executing locally ona can provide commit_id or may ommit it. Assigning default value for local executions:
    commit_id = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", default="unrecognised_commit_id")
    LOGGER.info(f"Images tested: {dlc_images}")
    all_image_list = dlc_images.split(" ")
    standard_images_list = [image_uri for image_uri in all_image_list if "example" not in image_uri]
    # Do not create EKS cluster for when EIA Only Images are present
    is_all_images_list_eia = all("eia" in image_uri for image_uri in all_image_list)
    eks_cluster_name = None
    benchmark_mode = "benchmark" in test_type or is_benchmark_dev_context()
    specific_test_type = (
        re.sub("benchmark-", "", test_type) if "benchmark" in test_type else test_type
    )
    build_context = get_build_context()

    # quick_checks tests don't have images in it. Using a placeholder here for jobs like that
    try:
        framework, version = get_framework_and_version_from_tag(all_image_list[0])
    except:
        framework, version = "general_test", "none"

    pytest_cache_params = {
        "codebuild_project_name": get_codebuild_project_name(),
        "commit_id": commit_id,
        "framework": generate_unique_dlc_name(dlc_images[0]),
        "version": version,
        "build_context": build_context,
        "test_type": test_type,
    }

    # In PR context, allow us to switch sagemaker tests to RC tests.
    # Do not allow them to be both enabled due to capacity issues.
    if specific_test_type == "sagemaker" and is_rc_test_context() and is_pr_context():
        specific_test_type = "release_candidate_integration"

    test_path = (
        os.path.join("benchmark", specific_test_type) if benchmark_mode else specific_test_type
    )

    # Skipping non HuggingFace/AG specific tests to execute only sagemaker tests
    is_hf_image_present = any("huggingface" in image_uri for image_uri in all_image_list)
    is_ag_image_present = any("autogluon" in image_uri for image_uri in all_image_list)
    is_trcomp_image_present = any("trcomp" in image_uri for image_uri in all_image_list)
    is_hf_image_present = is_hf_image_present and not is_trcomp_image_present
    is_hf_trcomp_image_present = is_hf_image_present and is_trcomp_image_present
    if (
        (is_hf_image_present or is_ag_image_present)
        and specific_test_type in ("ecs", "ec2", "eks", "bai")
    ) or (
        is_hf_trcomp_image_present
        and (
            specific_test_type in ("ecs", "eks", "bai", "release_candidate_integration")
            or benchmark_mode
        )
    ):
        # Creating an empty file for because codebuild job fails without it
        LOGGER.info(
            f"NOTE: {specific_test_type} tests not supported on HF, AG or Trcomp. Skipping..."
        )
        report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
        sm_utils.generate_empty_report(report, test_type, "huggingface")
        return

    if specific_test_type in (
        "sanity",
        "ecs",
        "ec2",
        "eks",
        "canary",
        "bai",
        "quick_checks",
        "release_candidate_integration",
    ):
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
        if specific_test_type == "bai":
            build_bai_docker_container()
        if specific_test_type == "eks" and not is_all_images_list_eia:
            frameworks_in_images = [
                framework
                for framework in ("mxnet", "pytorch", "tensorflow")
                if framework in dlc_images
            ]
            if len(frameworks_in_images) != 1:
                raise ValueError(
                    f"All images in dlc_images must be of a single framework for EKS tests.\n"
                    f"Instead seeing {frameworks_in_images} frameworks."
                )
            framework = frameworks_in_images[0]
            eks_cluster_name = f"dlc-{framework}-{build_context}"
            eks_utils.eks_setup()
            if eks_utils.is_eks_cluster_active(eks_cluster_name):
                eks_utils.eks_write_kubeconfig(eks_cluster_name)
            else:
                raise Exception(f"EKS cluster {eks_cluster_name} is not in active state")

        # Execute dlc_tests pytest command
        pytest_cmd = ["-s", "-rA", test_path, f"--junitxml={report}", "-n=auto"]

        is_habana_image = any("habana" in image_uri for image_uri in all_image_list)
        if specific_test_type == "ec2":
            if is_habana_image:
                context = Context()
                context.run("git clone https://github.com/HabanaAI/gaudi-test-suite.git")
                context.run("tar -c -f gaudi-test-suite.tar.gz gaudi-test-suite")
            else:
                pytest_cmd += ["--reruns=1", "--reruns-delay=10"]

        if is_pr_context():
            if specific_test_type == "eks":
                pytest_cmd.append("--timeout=2340")
            else:
                if is_habana_image:
                    pytest_cmd.append("--timeout=18000")
                else:
                    pytest_cmd.append("--timeout=4860")

        pytest_cmds = [pytest_cmd]
        # Execute separate cmd for canaries
        if specific_test_type in ("canary", "quick_checks"):
            pytest_cmds = [
                [
                    "-s",
                    "-rA",
                    f"--junitxml={report}",
                    "-n=auto",
                    f"--{specific_test_type}",
                    "--ignore=container_tests/",
                ]
            ]
            if specific_test_type == "canary":
                # Add rerun flag to canaries to avoid flakiness
                pytest_cmds = [
                    pytest_cmd + ["--reruns=1", "--reruns-delay=10"] for pytest_cmd in pytest_cmds
                ]

        pytest_cmds = [
            pytest_cmd + ["--last-failed", "--last-failed-no-failures", "all"]
            for pytest_cmd in pytest_cmds
        ]
        pytest_cache_util.download_pytest_cache_from_s3_to_local(os.getcwd(), **pytest_cache_params)
        try:
            # Note:- Running multiple pytest_cmds in a sequence will result in the execution log having two
            #        separate pytest reports, both of which must be examined in case of a manual review of results.
            cmd_exit_statuses = [pytest.main(pytest_cmd) for pytest_cmd in pytest_cmds]
            if all([status == 0 for status in cmd_exit_statuses]):
                sys.exit(0)
            elif any([status != 0 for status in cmd_exit_statuses]) and is_nightly_context():
                LOGGER.warning("\nSuppressed Failed Nightly Tests")
                for index, status in enumerate(cmd_exit_statuses):
                    if status != 0:
                        LOGGER.warning(
                            f'"{pytest_cmds[index]}" tests failed. Status code: {status}'
                        )
                sys.exit(0)
            else:
                raise RuntimeError(pytest_cmds)
        finally:
            pytest_cache_util.upload_pytest_cache_from_local_to_s3(
                os.getcwd(), **pytest_cache_params
            )
            # Delete dangling EC2 KeyPairs
            if os.path.exists(KEYS_TO_DESTROY_FILE):
                delete_key_pairs(KEYS_TO_DESTROY_FILE)
    elif specific_test_type == "sagemaker":
        if "habana" in dlc_images:
            LOGGER.info(f"Skipping SM tests for Habana. Images: {dlc_images}")
            # Creating an empty file for because codebuild job fails without it
            report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
            sm_utils.generate_empty_report(report, test_type, "habana")
            return
        if benchmark_mode:
            if "neuron" in dlc_images:
                LOGGER.info(f"Skipping benchmark sm tests for Neuron. Images: {dlc_images}")
                # Creating an empty file for because codebuild job fails without it
                report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
                sm_utils.generate_empty_report(report, test_type, "neuron")
                return
            report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
            os.chdir(os.path.join("test", "dlc_tests"))

            setup_sm_benchmark_env(dlc_images, test_path)
            pytest_cmd = [
                "-s",
                "-rA",
                test_path,
                f"--junitxml={report}",
                "-n=auto",
                "-o",
                "norecursedirs=resources",
            ]
            if not is_pr_context():
                pytest_cmd += ["--efa"] if efa_dedicated else ["-m", "not efa"]
            status = pytest.main(pytest_cmd)
            if is_nightly_context() and status != 0:
                LOGGER.warning("\nSuppressed Failed Nightly Tests")
                LOGGER.warning(f'"{pytest_cmd}" tests failed. Status code: {status}')
                sys.exit(0)
            else:
                sys.exit(status)

        else:
            sm_remote_images = [
                image
                for image in standard_images_list
                if not (("tensorflow-inference" in image and "py2" in image) or is_ec2_image(image))
            ]
            run_sagemaker_remote_tests(sm_remote_images, pytest_cache_params)
            if standard_images_list and not sm_remote_images:
                report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
                sm_utils.generate_empty_report(report, test_type, "sm_remote_unsupported")
        metrics_utils.send_test_duration_metrics(start_time)

    elif specific_test_type == "sagemaker-local":
        sm_local_to_skip = {
            "habana": "Skipping SM tests because SM does not yet support Habana",
            "neuron": "Skipping - there are no local mode tests for Neuron",
            "huggingface-tensorflow-training": "Skipping - there are no local mode tests for HF TF training",
        }

        for skip_condition, reason in sm_local_to_skip.items():
            if skip_condition in dlc_images:
                LOGGER.info(f"{reason}. Images: {dlc_images}")
                # Creating an empty file for because codebuild job fails without it
                report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
                sm_utils.generate_empty_report(report, test_type, skip_condition)
                return

        testing_image_list = [
            image
            for image in standard_images_list
            if not (
                ("tensorflow-inference" in image and "py2" in image)
                or ("eia" in image)
                or (is_ec2_image(image))
            )
        ]
        run_sagemaker_local_tests(testing_image_list, pytest_cache_params)
        # for EIA Images
        if len(testing_image_list) == 0:
            report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")
            sm_utils.generate_empty_report(report, test_type, "eia")
    else:
        raise NotImplementedError(
            f"{test_type} test is not supported. Only support ec2, ecs, eks, sagemaker and sanity currently"
        )


if __name__ == "__main__":
    main()
