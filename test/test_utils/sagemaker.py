import datetime
import os
import subprocess
import random
import re
import boto3
from botocore.config import Config
from time import sleep

import invoke
from invoke.context import Context
from invoke import exceptions
from junit_xml import TestSuite, TestCase

from test_utils import ec2 as ec2_utils
from test_utils import metrics as metrics_utils
from test_utils import (
    destroy_ssh_keypair,
    generate_ssh_keypair,
    get_framework_and_version_from_tag,
    get_job_type_from_image,
    get_python_invoker,
    is_pr_context,
    is_nightly_context,
    LOGGER,
    SAGEMAKER_EXECUTION_REGIONS,
    UBUNTU_18_BASE_DLAMI_US_EAST_1,
    UBUNTU_18_BASE_DLAMI_US_WEST_2,
    SAGEMAKER_LOCAL_TEST_TYPE,
    SAGEMAKER_REMOTE_TEST_TYPE,
    UBUNTU_HOME_DIR,
    DEFAULT_REGION,
)
from test_utils.pytest_cache import PytestCache


class DLCSageMakerRemoteTestFailure(Exception):
    pass


class DLCSageMakerLocalTestFailure(Exception):
    pass


def assign_sagemaker_remote_job_instance_type(image):
    if "neuron" in image:
        return "ml.inf1.xlarge"
    elif "gpu" in image:
        return "ml.p3.8xlarge"
    elif "tensorflow" in image:
        return "ml.c4.4xlarge"
    else:
        return "ml.c4.8xlarge"


def assign_sagemaker_local_job_instance_type(image):
    if "tensorflow" in image and "inference" in image and "gpu" in image:
        return "p2.xlarge"
    elif "autogluon" in image and "gpu" in image:
        return "p3.2xlarge"
    return "p3.8xlarge" if "gpu" in image else "c5.18xlarge"


def launch_sagemaker_local_ec2_instance(image, ami_id, ec2_key_name, region):
    """
    Launch Ec2 instance for running sagemaker local tests
    :param image: str
    :param ami_id: str
    :param ec2_key_name: str
    :param region: str
    :return: str, str
    """
    instance_type = assign_sagemaker_local_job_instance_type(image)
    instance_name = image.split("/")[-1]
    instance = ec2_utils.launch_instance(
        ami_id,
        region=region,
        ec2_key_name=ec2_key_name,
        instance_type=instance_type,
        # EIA does not have SM Local test
        ei_accelerator_type=None,
        user_data=None,
        iam_instance_profile_name=ec2_utils.EC2_INSTANCE_ROLE_NAME,
        instance_name=f"sm-local-{instance_name}",
    )
    instance_id = instance["InstanceId"]
    public_ip_address = ec2_utils.get_public_ip(instance_id, region=region)
    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(instance_id, system_status="ok", instance_status="ok", region=region)
    return instance_id, public_ip_address


def generate_sagemaker_pytest_cmd(image, sagemaker_test_type):
    """
    Parses the image ECR url and returns appropriate pytest command
    :param image: ECR url of image
    :param sagemaker_test_type: local or remote test type
    :return: <tuple> pytest command to be run, path where it should be executed, image tag
    """
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    print("image name {}".format(image))
    sm_remote_docker_base_name, tag = image.split("/")[1].split(":")
    sm_local_docker_repo_uri = image.split(":")[0]

    # Assign instance type
    instance_type = assign_sagemaker_remote_job_instance_type(image)

    # Get path to test directory
    find_path = sm_remote_docker_base_name.split("-")

    # NOTE: We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework, framework_version = get_framework_and_version_from_tag(image)
    framework_major_version = framework_version.split(".")[0]
    job_type = get_job_type_from_image(image)
    path = os.path.join("test", "sagemaker_tests", framework, job_type)
    aws_id_arg = "--aws-id"
    docker_base_arg = "--docker-base-name"
    instance_type_arg = "--instance-type"
    accelerator_type_arg = "--accelerator-type"
    framework_version_arg = "--framework-version"
    eia_arg = "ml.eia1.large"
    processor = (
        "neuron"
        if "neuron" in image
        else "gpu"
        if "gpu" in image
        else "eia"
        if "eia" in image
        else "cpu"
    )
    py_version = re.search(r"py\d+", tag).group()
    sm_local_py_version = "37" if py_version == "py37" else "38" if py_version == "py38" else "2" if py_version == "py27" else "3"
    if framework == "tensorflow" and job_type == "inference":
        # Tf Inference tests have an additional sub directory with test
        integration_path = os.path.join("test", "integration", sagemaker_test_type)
    else:
        integration_path = os.path.join("integration", sagemaker_test_type)

    # Conditions for modifying tensorflow SageMaker pytest commands
    if framework == "tensorflow" and sagemaker_test_type == SAGEMAKER_REMOTE_TEST_TYPE:
        if job_type == "inference":
            aws_id_arg = "--registry"
            docker_base_arg = "--repo"
            instance_type_arg = "--instance-types"
            framework_version_arg = "--versions"
            integration_path = os.path.join(integration_path, "test_tfs.py") if processor != "eia" else os.path.join(
                integration_path, "test_ei.py")

    if framework == "tensorflow" and job_type == "training":
        aws_id_arg = "--account-id"

    test_report = os.path.join(os.getcwd(), "test", f"{job_type}_{tag}.xml")
    local_test_report = os.path.join(UBUNTU_HOME_DIR, "test", f"{job_type}_{tag}_sm_local.xml")

    # Explanation of why we need the if-condition below:
    # We have separate Pipeline Actions that run EFA tests, which have the env variable "EFA_DEDICATED=True" configured
    # so that those Actions only run the EFA tests.
    # However, there is no such dedicated CB job dedicated to EFA tests in the PR context. This means that when in the
    # PR context, setting "DISABLE_EFA_TESTS" to True should skip EFA tests, but setting it to False should enable
    # not just the EFA tests, but also all other tests as well.
    if is_pr_context():
        efa_tests_disabled = os.getenv("DISABLE_EFA_TESTS", "False").lower() == "true"
        efa_flag = "-m \"not efa\"" if efa_tests_disabled else ""
    else:
        efa_dedicated = os.getenv("EFA_DEDICATED", "False").lower() == "true"
        efa_flag = '--efa' if efa_dedicated else '-m \"not efa\"'

    region_list = ",".join(SAGEMAKER_EXECUTION_REGIONS)

    sagemaker_regions_list = f"--sagemaker-regions {region_list}"

    remote_pytest_cmd = (
        f"pytest -rA {integration_path} --region {region} --processor {processor} {docker_base_arg} "
        f"{sm_remote_docker_base_name} --tag {tag} {framework_version_arg} {framework_version} "
        f"{aws_id_arg} {account_id} {instance_type_arg} {instance_type} {efa_flag} {sagemaker_regions_list} --junitxml {test_report}"
    )

    if processor == "eia":
        remote_pytest_cmd += f"{accelerator_type_arg} {eia_arg}"

    local_pytest_cmd = (f"pytest -s -v {integration_path} {docker_base_arg} "
                        f"{sm_local_docker_repo_uri} --tag {tag} --framework-version {framework_version} "
                        f"--processor {processor} {aws_id_arg} {account_id} --junitxml {local_test_report}")

    if framework == "tensorflow" and job_type != "inference":
        local_pytest_cmd = f"{local_pytest_cmd} --py-version {sm_local_py_version} --region {region}"
    if framework == "tensorflow" and job_type == "training":
        path = os.path.join(os.path.dirname(path), f"{framework}{framework_major_version}_training")
    if "huggingface" in framework and job_type == "inference":
        path = os.path.join("test", "sagemaker_tests", "huggingface", "inference")

    return (
        remote_pytest_cmd if sagemaker_test_type == SAGEMAKER_REMOTE_TEST_TYPE else local_pytest_cmd,
        path,
        tag,
        job_type,
    )


def install_sm_local_dependencies(framework, job_type, image, ec2_conn, ec2_instance_ami):
    """
    Install sagemaker local test dependencies
    :param framework: str
    :param job_type: str
    :param image: str
    :param ec2_conn: Fabric_obj
    :return: None
    """
    python_invoker = get_python_invoker(ec2_instance_ami)
    # Install custom packages which need to be latest version"
    # using virtualenv to avoid package conflicts with the current packages
    ec2_conn.run(f"sudo apt-get install virtualenv -y ")
    ec2_conn.run(f"virtualenv env --python {python_invoker}")
    ec2_conn.run(f"source ./env/bin/activate")
    if framework == "pytorch":
        # The following distutils package conflict with test dependencies
        ec2_conn.run("sudo apt-get remove python3-scipy python3-yaml -y")
    ec2_conn.run(f"sudo {python_invoker} -m pip install -r requirements.txt ", warn=True)


def kill_background_processes_and_run_apt_get_update(ec2_conn):
    """
    The apt-daily services on the DLAMI cause a conflict upon running any "apt install" commands within the first few
    minutes of starting an EC2 instance. These services are not necessary for the purpose of the DLC tests, and can
    therefore be killed. This function kills the services, and then forces "apt-get update" to run in the foreground.

    :param ec2_conn: Fabric SSH connection
    :return:
    """
    apt_daily_services_list = ["apt-daily.service", "apt-daily-upgrade.service", "unattended-upgrades.service"]
    apt_daily_services = " ".join(apt_daily_services_list)
    ec2_conn.run(f"sudo systemctl stop {apt_daily_services}")
    ec2_conn.run(f"sudo systemctl kill --kill-who=all {apt_daily_services}")
    num_stopped_services = 0
    # The `systemctl kill` command is expected to take about 1 second. The 60 second loop here exists to force
    # the execution to wait (if needed) for a longer amount of time than it would normally take to kill the services.
    for _ in range(60):
        sleep(1)
        # List the apt-daily services, get the number of dead services
        num_stopped_services = int(ec2_conn.run(
            f"systemctl list-units --all {apt_daily_services} | egrep '(dead|failed)' | wc -l"
        ).stdout.strip())
        # Exit condition for the loop is when all apt daily services are dead.
        if num_stopped_services == len(apt_daily_services_list):
            break
    if num_stopped_services != len(apt_daily_services_list):
        raise RuntimeError(
            "Failed to kill background services to allow apt installs on SM Local EC2 instance. "
            f"{len(apt_daily_services) - num_stopped_services} still remaining."
        )
    ec2_conn.run("sudo rm -rf /var/lib/dpkg/lock*;")
    ec2_conn.run("sudo dpkg --configure -a;")
    ec2_conn.run("sudo apt-get update")
    return


def execute_local_tests(image, pytest_cache_params):
    """
    Run the sagemaker local tests in ec2 instance for the image
    :param image: ECR url
    :param pytest_cache_params: parameters required for :param pytest_cache_util
    :return: None
    """
    account_id = os.getenv("ACCOUNT_ID", boto3.client("sts").get_caller_identity()["Account"])
    pytest_cache_util = PytestCache(boto3.client("s3"), account_id)
    ec2_client = boto3.client("ec2", config=Config(retries={"max_attempts": 10}), region_name=DEFAULT_REGION)
    pytest_command, path, tag, job_type = generate_sagemaker_pytest_cmd(image, SAGEMAKER_LOCAL_TEST_TYPE)
    pytest_command += " --last-failed --last-failed-no-failures all "
    print(pytest_command)
    framework, _ = get_framework_and_version_from_tag(image)
    random.seed(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    ec2_key_name = f"{job_type}_{tag}_sagemaker_{random.randint(1, 1000)}"
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
    ec2_ami_id = UBUNTU_18_BASE_DLAMI_US_EAST_1 if region == "us-east-1" else UBUNTU_18_BASE_DLAMI_US_WEST_2
    sm_tests_tar_name = "sagemaker_tests.tar.gz"
    ec2_test_report_path = os.path.join(UBUNTU_HOME_DIR, "test", f"{job_type}_{tag}_sm_local.xml")
    instance_id = ""
    ec2_conn = None
    try:
        key_file = generate_ssh_keypair(ec2_client, ec2_key_name)
        print(f"Launching new Instance for image: {image}")
        instance_id, ip_address = launch_sagemaker_local_ec2_instance(
            image,
            ec2_ami_id,
            ec2_key_name,
            region
        )
        ec2_conn = ec2_utils.get_ec2_fabric_connection(instance_id, key_file, region)
        ec2_conn.put(sm_tests_tar_name, f"{UBUNTU_HOME_DIR}")
        ec2_conn.run(f"$(aws ecr get-login --no-include-email --region {region})")
        try:
            ec2_conn.run(f"docker pull {image}", timeout=600)
        except invoke.exceptions.CommandTimedOut as e:
            output = ec2_conn.run(f"docker images {image} --format '{{.Repository}}:{{.Tag}}'").stdout.strip("\n")
            if output != image:
                raise DLCSageMakerLocalTestFailure(
                    f"Image pull for {image} failed.\ndocker images output = {output}"
                ) from e
        ec2_conn.run(f"tar -xzf {sm_tests_tar_name}")
        kill_background_processes_and_run_apt_get_update(ec2_conn)
        with ec2_conn.cd(path):
            install_sm_local_dependencies(framework, job_type, image, ec2_conn, ec2_ami_id)
            pytest_cache_util.download_pytest_cache_from_s3_to_ec2(ec2_conn, path, **pytest_cache_params)
            # Workaround for mxnet cpu training images as test distributed
            # causes an issue with fabric ec2_connection
            if framework == "mxnet" and job_type == "training" and "cpu" in image:
                try:
                    ec2_conn.run(pytest_command, timeout=1000, warn=True)
                except exceptions.CommandTimedOut as exc:
                    print(f"Ec2 connection timed out for {image}, {exc}")
                finally:
                    print(f"Downloading Test reports for image: {image}")
                    ec2_conn.close()
                    ec2_conn_new = ec2_utils.get_ec2_fabric_connection(instance_id, key_file, region)
                    ec2_conn_new.get(ec2_test_report_path,
                                     os.path.join("test", f"{job_type}_{tag}_sm_local.xml"))
                    output = subprocess.check_output(f"cat test/{job_type}_{tag}_sm_local.xml", shell=True,
                                                     executable="/bin/bash")
                    pytest_cache_util.upload_pytest_cache_from_ec2_to_s3(ec2_conn_new, path, **pytest_cache_params)
                    if 'failures="0"' not in str(output):
                        error_message = f"Sagemaker Local tests failed for {image}"
                        if is_nightly_context():
                            LOGGER.warning(error_message)
                        raise ValueError(error_message)
            else:
                ec2_conn.run(pytest_command)
                print(f"Downloading Test reports for image: {image}")
                ec2_conn.get(ec2_test_report_path, os.path.join("test", f"{job_type}_{tag}_sm_local.xml"))
    finally:
        with ec2_conn.cd(path):
            pytest_cache_util.upload_pytest_cache_from_ec2_to_s3(ec2_conn, path, **pytest_cache_params)
        print(f"Terminating Instances for image: {image}")
        ec2_utils.terminate_instance(instance_id, region)
        print(f"Destroying ssh Key_pair for image: {image}")
        destroy_ssh_keypair(ec2_client, ec2_key_name)
        # return None here to prevent errors from multiprocessing.map(). Without this it returns some object by default
        # which is causing "cannot pickle '_thread.lock' object" error
        return None


def execute_sagemaker_remote_tests(process_index, image, global_pytest_cache, pytest_cache_params):
    """
    Run pytest in a virtual env for a particular image. Creates a custom directory for each thread for pytest cache file.
    Stores pytest cache in a shared dict.
    Expected to run via multiprocessing
    :param process_index - id for process. Used to create a custom cache dir
    :param image - ECR url
    :param global_pytest_cache - shared Manager().dict() for cache merging
    :param pytest_cache_params - parameters required for s3 file path building
    """
    account_id = os.getenv("ACCOUNT_ID", boto3.client("sts").get_caller_identity()["Account"])
    pytest_cache_util = PytestCache(boto3.client("s3"), account_id)
    pytest_command, path, tag, job_type = generate_sagemaker_pytest_cmd(image, SAGEMAKER_REMOTE_TEST_TYPE)
    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            pytest_cache_util.download_pytest_cache_from_s3_to_local(path, **pytest_cache_params, custom_cache_directory=str(process_index))
            # adding -o cache_dir with a custom directory name
            pytest_command += f" -o cache_dir={os.path.join(str(process_index), '.pytest_cache')}"
            res = context.run(pytest_command, warn=True)
            metrics_utils.send_test_result_metrics(res.return_code)
            cache_json = pytest_cache_util.convert_pytest_cache_file_to_json(path, custom_cache_directory=str(process_index))
            global_pytest_cache.update(cache_json)
            if res.failed:
                error_message = (
                    f"{pytest_command} failed with error code: {res.return_code}\n"
                    f"Traceback:\n{res.stdout}"
                )
                if is_nightly_context():
                    LOGGER.warning(error_message)
                else:
                    raise DLCSageMakerRemoteTestFailure(error_message)
    return None


def generate_empty_report(report, test_type, case):
    """
    Generate empty junitxml report if no tests are run
    :param report: CodeBuild Report
    Returns: None
    """
    test_cases = [TestCase(test_type, case, 1, f"Skipped {test_type} on {case}", '')]
    ts = TestSuite(report, test_cases)
    with open(report, "w") as skip_file:
        TestSuite.to_file(skip_file, [ts], prettyprint=False)
