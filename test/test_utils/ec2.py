import os
import time
import re
from inspect import signature
import boto3

from retrying import retry
from fabric import Connection
from botocore.config import Config
from botocore.exceptions import ClientError

from test.test_utils import is_pr_context
from . import DEFAULT_REGION, UL_AMI_LIST, LOGGER, BENCHMARK_RESULTS_S3_BUCKET

EC2_INSTANCE_ROLE_NAME = "ec2TestInstanceRole"

# List of instance types for which if instance spin-up fails, the test is skipped instead of failing.
ICE_SKIP_INSTANCE_LIST = ["p3dn.24xlarge"]


def get_ec2_instance_type(default, processor, disable_p3dn=False):
    """
    Get EC2 instance type from associated EC2_[CPU|GPU]_INSTANCE_TYPE env variable, or set it to a default
    for contexts where the variable is not present (i.e. PR, Nightly, local testing)

    :param default: Default instance type to use - Should never be p3dn
    :param processor: "cpu" or "gpu"
    :param disable_p3dn: Boolean to determine whether or not to run tests on p3dn. If set to true, default
    gpu instance type will be used.

    :return: one item list of instance type -- this is used to parametrize tests, and parameter is required to be
    a list.
    """
    allowed_processors = ("cpu", "gpu")
    p3dn = "p3dn.24xlarge"
    p28x = "p2.8xlarge"
    if processor not in allowed_processors:
        raise RuntimeError(
            f"Aborting EC2 test run. Unrecognized processor type {processor}. "
            f"Please choose from {allowed_processors}"
        )
    if default == p3dn:
        raise RuntimeError("Default instance type should never be p3dn.24xlarge")
    instance_type = os.getenv(f"EC2_{processor.upper()}_INSTANCE_TYPE", default)

    # TODO: Re-enable p28x once capacity issues have been resolved
    if instance_type == p28x:
        return []
    if instance_type == p3dn and disable_p3dn:
        instance_type = default
    return [instance_type]


def get_ec2_accelerator_type(default, processor):
    """
    Get EC2 instance type from associated EC2_[CPU|GPU]_INSTANCE_TYPE env variable, or set it to a default
    for contexts where the variable is not present (i.e. PR, Nightly, local testing)

    :param default: Default instance type to use - Should never be p3dn
    :param processor: "eia"

    :return: one item list of instance type -- this is used to parametrize tests, and parameter is required to be
    a list.
    """
    allowed_processors = ("eia", "neuron")
    if processor not in allowed_processors:
        raise RuntimeError(
            f"Aborting EC2 test run. Unrecognized processor type {processor}. "
            f"Please choose from {allowed_processors}"
        )
    accelerator_type = os.getenv(f"EC2_{processor.upper()}_INSTANCE_TYPE", default)
    return [accelerator_type]


def launch_instance(
    ami_id,
    instance_type,
    ei_accelerator_type,
    ec2_key_name=None,
    region=DEFAULT_REGION,
    user_data=None,
    iam_instance_profile_name=None,
    instance_name="",
):
    """
    Launch an instance
    :param ami_id: AMI ID to be used for launched instance
    :param instance_type: Instance type of launched instance
    :param region: Region where instance will be launched
    :param user_data: Script to run when instance is launched as a str
    :param iam_instance_profile_arn: EC2 Role to be attached
    :param instance_name: Tag to display as Name on EC2 Console
    :return: <dict> Information about the instance that was launched
    """
    if not ami_id:
        raise Exception("No ami_id provided")
    if not ec2_key_name:
        raise Exception("Ec2 Key name must be provided")
    client = boto3.Session(region_name=region).client("ec2")

    # Construct the dictionary with the arguments for API call
    arguments_dict = {
        "KeyName": ec2_key_name,
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MaxCount": 1,
        "MinCount": 1,
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"CI-CD {instance_name}"}],},
        ],
    }
    if user_data:
        arguments_dict["UserData"] = user_data
    if iam_instance_profile_name:
        arguments_dict["IamInstanceProfile"] = {"Name": iam_instance_profile_name}
    if ei_accelerator_type:
        arguments_dict["ElasticInferenceAccelerators"] = ei_accelerator_type
        availability_zones = {
            "us-west": ["us-west-2a", "us-west-2b", "us-west-2c"],
            "us-east": ["us-east-1a", "us-east-1b", "us-east-1c"],
        }
        for a_zone in availability_zones[region]:
            arguments_dict["Placement"] = {"AvailabilityZone": a_zone}
            try:
                response = client.run_instances(**arguments_dict)
                if response and len(response["Instances"]) >= 1:
                    break
            except ClientError as e:
                print(f"Failed to launch in {a_zone} with Error: {e}")
                continue
    else:
        response = client.run_instances(**arguments_dict)

    if not response or len(response["Instances"]) < 1:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any response"
        )

    return response["Instances"][0]


def get_ec2_client(region):
    return boto3.client("ec2", region_name=region, config=Config(retries={"max_attempts": 10}))


def get_instance_from_id(instance_id, region=DEFAULT_REGION):
    """
    Get instance information using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client("ec2")
    instance = client.describe_instances(InstanceIds=[instance_id])
    if not instance:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any reservations object"
        )
    return instance["Reservations"][0]["Instances"][0]


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def get_public_ip(instance_id, region=DEFAULT_REGION):
    """
    Get Public IP of instance using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> IP Address of instance with matching instance ID
    """
    instance = get_instance_from_id(instance_id, region)
    if not instance["PublicIpAddress"]:
        raise Exception("IP address not yet available")
    return instance["PublicIpAddress"]


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def get_public_ip_from_private_dns(private_dns, region=DEFAULT_REGION):
    """
    Get Public IP of instance using private DNS
    :param private_dns:
    :param region:
    :return: <str> IP Address of instance with matching private DNS
    """
    client = boto3.Session(region_name=region).client("ec2")
    response = client.describe_instances(Filters={"Name": "private-dns-name", "Value": [private_dns]})
    return response.get("Reservations")[0].get("Instances")[0].get("PublicIpAddress")


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def get_instance_user(instance_id, region=DEFAULT_REGION):
    """
    Get "ubuntu" or "ec2-user" based on AMI used to launch instance
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> user name
    """
    instance = get_instance_from_id(instance_id, region)
    user = "ubuntu" if instance["ImageId"] in UL_AMI_LIST else "ec2-user"
    return user


def get_instance_state(instance_id, region=DEFAULT_REGION):
    """
    Get state of instance using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> State of instance with matching instance ID
    """
    instance = get_instance_from_id(instance_id, region)
    return instance["State"]["Name"]


@retry(stop_max_attempt_number=16, wait_fixed=60000)
def check_instance_state(instance_id, state="running", region=DEFAULT_REGION):
    """
    Compares the instance state with the state argument.
    Retries 8 times with 120 seconds gap between retries.
    :param instance_id: Instance ID to be queried
    :param state: Expected instance state
    :param region: Region where query will be performed
    :return: <str> State of instance with matching instance ID
    """
    instance_state = get_instance_state(instance_id, region)
    if state != instance_state:
        raise Exception(f"Instance {instance_id} not in {state} state")
    return instance_state


def get_system_state(instance_id, region=DEFAULT_REGION):
    """
    Returns health checks state for instances
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <tuple> System state and Instance state of instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client("ec2")
    response = client.describe_instance_status(InstanceIds=[instance_id])
    if not response:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any reservations object"
        )
    instance_status_list = response["InstanceStatuses"]
    if not instance_status_list:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any reservations object"
        )
    if len(instance_status_list) < 1:
        raise Exception(
            "The instance id seems to be incorrect {}. \
                         reservations seems to be empty".format(
                instance_id
            )
        )

    instance_status = instance_status_list[0]
    return (
        instance_status["SystemStatus"]["Status"],
        instance_status["InstanceStatus"]["Status"],
    )


@retry(stop_max_attempt_number=96, wait_fixed=10000)
def check_system_state(instance_id, system_status="ok", instance_status="ok", region=DEFAULT_REGION):
    """
    Compares the system state (Health Checks).
    Retries 96 times with 10 seconds gap between retries
    :param instance_id: Instance ID to be queried
    :param system_status: Expected system state
    :param instance_status: Expected instance state
    :param region: Region where query will be performed
    :return: <tuple> System state and Instance state of instance with matching instance ID
    """
    instance_state = get_system_state(instance_id, region=region)
    if system_status != instance_state[0] or instance_status != instance_state[1]:
        raise Exception(
            "Instance {} not in \
                         required state".format(
                instance_id
            )
        )
    return instance_state


def terminate_instance(instance_id, region=DEFAULT_REGION):
    """
    Terminate EC2 instances with matching instance ID
    :param instance_id: Instance ID to be terminated
    :param region: Region where instance is located
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client("ec2")
    response = client.terminate_instances(InstanceIds=[instance_id])
    if not response:
        raise Exception("Unable to terminate instance. No response received.")
    instances_terminated = response["TerminatingInstances"]
    if not instances_terminated:
        raise Exception("Failed to terminate instance.")
    if instances_terminated[0]["InstanceId"] != instance_id:
        raise Exception("Failed to terminate instance. Unknown error.")


def get_instance_type_details(instance_type, region=DEFAULT_REGION):
    """
    Get instance type details for a given instance type
    :param instance_type: Instance type to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance type
    """
    client = boto3.client("ec2", region_name=region)
    response = client.describe_instance_types(InstanceTypes=[instance_type])
    if not response or not response["InstanceTypes"]:
        raise Exception("Unable to get instance details. No response received.")
    if response["InstanceTypes"][0]["InstanceType"] != instance_type:
        raise Exception(
            f"Bad response received. Requested {instance_type} "
            f"but got {response['InstanceTypes'][0]['InstanceType']}"
        )
    return response["InstanceTypes"][0]


def get_instance_details(instance_id, region=DEFAULT_REGION):
    """
    Get instance details for instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    instance = get_instance_from_id(instance_id, region=region)
    if not instance:
        raise Exception("Could not find instance")

    return get_instance_type_details(instance["InstanceType"], region=region)


@retry(stop_max_attempt_number=30, wait_fixed=10000)
def get_instance_num_cpus(instance_id, region=DEFAULT_REGION):
    """
    Get number of VCPUs on instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <int> Number of VCPUs on instance with matching instance ID
    """
    instance_info = get_instance_details(instance_id, region=region)
    return instance_info["VCpuInfo"]["DefaultVCpus"]


@retry(stop_max_attempt_number=30, wait_fixed=10000)
def get_instance_memory(instance_id, region=DEFAULT_REGION):
    """
    Get total RAM available on instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <int> Total RAM available on instance with matching instance ID
    """
    instance_info = get_instance_details(instance_id, region=region)
    return instance_info["MemoryInfo"]["SizeInMiB"]


@retry(stop_max_attempt_number=30, wait_fixed=10000)
def get_instance_num_gpus(instance_id=None, instance_type=None, region=DEFAULT_REGION):
    """
    Get total number of GPUs on instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param instance_type: Instance Type to be queried
    :param region: Region where query will be performed
    :return: <int> Number of GPUs on instance with matching instance ID
    """
    assert instance_id or instance_type, "Input must be either instance_id or instance_type"
    instance_info = (
        get_instance_type_details(instance_type, region=region)
        if instance_type
        else get_instance_details(instance_id, region=region)
    )
    return sum(gpu_type["Count"] for gpu_type in instance_info["GpuInfo"]["Gpus"])


def get_ec2_fabric_connection(instance_id, instance_pem_file, region):
    """
    establish connection with EC2 instance if necessary
    :param instance_id: ec2_instance id
    :param instance_pem_file: instance key name
    :param region: Region where ec2 instance is launched
    :return: Fabric connection object
    """
    user = get_instance_user(instance_id, region=region)
    conn = Connection(
        user=user, host=get_public_ip(instance_id, region), connect_kwargs={"key_filename": [instance_pem_file]},
    )
    return conn


def execute_ec2_training_test(
    connection,
    ecr_uri,
    test_cmd,
    region=DEFAULT_REGION,
    executable="bash",
    large_shm=False,
    host_network=False,
    container_name="ec2_training_container",
    timeout=3000,
):
    if executable not in ("bash", "python"):
        raise RuntimeError(f"This function only supports executing bash or python commands on containers")
    if executable == "bash":
        executable = os.path.join(os.sep, "bin", "bash")
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    # Run training command
    shm_setting = '--shm-size="1g"' if large_shm else ""
    network = '--network="host" ' if host_network else ""
    connection.run(
        f"{docker_cmd} run --name {container_name} {network}-v {container_test_local_dir}:{os.path.join(os.sep, 'test')}"
        f" {shm_setting} -itd {ecr_uri}",
        hide=True,
    )
    return connection.run(
        f"{docker_cmd} exec --user root {container_name} {executable} -c '{test_cmd}'", hide=True, timeout=timeout,
    )


def execute_ec2_inference_test(connection, ecr_uri, test_cmd, region=DEFAULT_REGION):
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    # Run training command
    connection.run(
        f"{docker_cmd} run --name ec2_inference_container -v {container_test_local_dir}:{os.path.join(os.sep, 'test')}"
        f" -itd {ecr_uri} bash",
        hide=True,
    )
    connection.run(
        f"{docker_cmd} exec --user root ec2_inference_container {os.path.join(os.sep, 'bin', 'bash')} -c '{test_cmd}'",
        hide=True,
        timeout=3000,
    )


def execute_ec2_training_performance_test(
    connection, ecr_uri, test_cmd, region=DEFAULT_REGION, post_process=None, data_source="", threshold=None,
):
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_name = f"{data_source}_results_{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}_{timestamp}.txt"
    log_location = os.path.join(container_test_local_dir, "benchmark", "logs", log_name)

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    connection.run(f"{docker_cmd} pull -q {ecr_uri}")

    # Run training command, display benchmark results to console
    connection.run(
        f"{docker_cmd} run --user root "
        f"-e LOG_FILE={os.path.join(os.sep, 'test', 'benchmark', 'logs', log_name)} "
        f"-e PR_CONTEXT={1 if is_pr_context() else 0} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {ecr_uri} "
        f"{os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}"
    )
    ec2_performance_upload_result_to_s3_and_validate(
        connection, ecr_uri, log_location, data_source, threshold, post_process, log_name,
    )


def execute_ec2_inference_performance_test(
    connection, ecr_uri, test_cmd, region=DEFAULT_REGION, post_process=None, data_source="", threshold=None,
):
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_name = f"{data_source}_results_{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}_{timestamp}.txt"
    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
    connection.run(f"{docker_cmd} pull -q {ecr_uri}")

    # Run training command, display benchmark results to console
    repo_name, image_tag = ecr_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    connection.run(
        f"{docker_cmd} run -d --name {container_name} "
        f"-e LOG_FILE={os.path.join(os.sep, 'test', 'benchmark', 'logs', log_name)} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {ecr_uri}"
    )
    try:
        connection.run(
            f"{docker_cmd} exec --user root {container_name} " f"{os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}"
        )
    except Exception as e:
        raise Exception("Failed to exec benchmark command.\n", e)
    finally:
        connection.run(f"docker rm -f {container_name}")
    log_location = os.path.join(container_test_local_dir, "benchmark", "logs", log_name)
    ec2_performance_upload_result_to_s3_and_validate(
        connection, ecr_uri, log_location, data_source, threshold, post_process, log_name,
    )


def ec2_performance_upload_result_to_s3_and_validate(
    connection, ecr_uri, log_location, data_source, threshold, post_process, log_name
):
    framework = "tensorflow" if "tensorflow" in ecr_uri else "mxnet" if "mxnet" in ecr_uri else "pytorch"
    framework_version = re.search(r"\d+(\.\d+){2}", ecr_uri).group()
    py_version = "py2" if "py2" in ecr_uri else "py37" if "py37" in ecr_uri else "py3"
    processor = "gpu" if "gpu" in ecr_uri else "cpu"
    work_type = "training" if "training" in ecr_uri else "inference"
    s3_location = os.path.join(
        BENCHMARK_RESULTS_S3_BUCKET, framework, framework_version, "ec2", work_type, processor, py_version, log_name,
    )
    params = {"connection": connection, "log_location": log_location}
    if "threshold" in signature(post_process).parameters:
        params["threshold"] = threshold
    performance_number = post_process(**params)
    unit = (
        "s"
        if work_type == "inference" and framework == "tensorflow"
        else "ms"
        if work_type == "inference" and framework == "pytorch"
        else "s/epoch"
        if work_type == "training" and framework == "pytorch" and data_source == "imagenet"
        else "images/sec"
    )
    description = "p99 latency " if unit == "s" or unit == "ms" else ""
    for k, v in performance_number.items():
        performance_statement = (
            f"{framework} {framework_version} ec2 {work_type} {processor} {py_version} "
            f"{data_source} {k} {description}: {v} {unit}, threshold: {threshold[k]} {unit}"
        )
        connection.run(f"echo {performance_statement} | sudo tee -a {log_location}")
        LOGGER.info(f"{performance_statement}")
    connection.run(f"aws s3 cp {log_location} {s3_location}")
    LOGGER.info(f"To retrieve complete benchmark log, check {s3_location}")

    def _assertion_results():
        if "Cost" in performance_number:
            return performance_number["Cost"] < threshold["Cost"]
        if "Throughput" in performance_number:
            return performance_number["Throughput"] > threshold["Throughput"]
        if len(performance_number) == 0:
            return False
        failure_count = 0
        for k, v in performance_number.items():
            if v > threshold[k]:
                failure_count += 1
        return failure_count <= 2

    for _ in performance_number:
        assert _assertion_results(), (
            f"{framework} {framework_version} ec2 {work_type} {processor} {py_version} {data_source} "
            f"Benchmark Result {performance_number} does not reach the threshold {threshold}"
        )


def post_process_inference(connection, log_location, threshold):
    log_content = connection.run(f"cat {log_location}").stdout.split("\n")
    performance_number = {}
    for line in log_content:
        if "p99" in line:
            for key in threshold.keys():
                if key in line:
                    performance_number[key] = float(
                        re.search(r"(p99[ ]*(Latency)?[ ]*:[ ]*)(?P<result>[0-9]+\.?[0-9]+)", line,).group("result")
                    )
                    break
    return performance_number


def post_process_mxnet_ec2_performance(connection, log_location):
    log_content = connection.run(f"cat {log_location}").stdout.split("\n")
    total = 0.0
    n = 0
    for line in log_content:
        if "samples/sec" in line and "warmup" not in line:
            throughput = re.search(r"((?P<throughput>[0-9]+\.?[0-9]+)[ ]+samples/sec)", line).group("throughput")
            total += float(throughput)
            n += 1
    if total and n:
        return {"Throughput": total / n}
    else:
        raise ValueError("total: {}; n: {} -- something went wrong".format(total, n))
