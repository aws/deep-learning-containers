import os

import boto3

from retrying import retry

from test.test_utils import DEFAULT_REGION, UBUNTU_16_BASE_DLAMI, LOGGER


EC2_INSTANCE_ROLE_NAME = "ec2TestInstanceRole"


def get_ec2_instance_type(default, processor, enable_p3dn=False):
    """
    Get EC2 instance type from associated EC2_[CPU|GPU]_INSTANCE_TYPE env variable, or set it to a default
    for contexts where the variable is not present (i.e. PR, Nightly, local testing)

    :param default: Default instance type to use
    :param processor: "cpu" or "gpu"
    :param enable_p3dn: Boolean to determine whether or not to run tests on p3dn. If set to false, default
    gpu instance type will be used. If default gpu instance type is p3dn, then use small gpu instance type (p2.xlarge)

    :return: one item list of instance type -- this is used to parametrize tests, and parameter is required to be
    a list.
    """
    allowed_processors = ("cpu", "gpu", "eia")
    p3dn = "p3dn.24xlarge"
    if processor not in allowed_processors:
        raise RuntimeError(
            f"Aborting EC2 test run. Unrecognized processor type {processor}. "
            f"Please choose from {allowed_processors}"
        )
    if default == p3dn and not enable_p3dn:
        raise RuntimeError(
            "Default instance type is p3dn but p3dn testing is disabled. Please either enable p3dn "
            "by setting enable_p3dn=True, or change the default instance type"
        )
    instance_type = os.getenv(f"EC2_{processor.upper()}_INSTANCE_TYPE", default)
    if instance_type == p3dn and not enable_p3dn:
        return [default]
    return [instance_type]


def launch_instance(
    ami_id, instance_type, region=DEFAULT_REGION, user_data=None, iam_instance_profile_arn=None, instance_name="", ei_accelerator_type=None,
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
    client = boto3.Session(region_name=region).client("ec2")

    # Construct the dictionary with the arguments for API call
    arguments_dict = {
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
    if iam_instance_profile_arn:
        arguments_dict["IamInstanceProfile"] = {"Arn": iam_instance_profile_arn}
    if ei_accelerator_type:
        arguments_dict["ElasticInferenceAccelerators"] = ei_accelerator_type
        availability_zones = {"us-west": ["us-west-2a", "us-west-2b", "us-west-2c"],
                              "us-east": ["us-east-1a", "us-east-1b", "us-east-1c"]}
        res = {}
        for a_zone in availability_zones[region]:
            arguments_dict["Placement"] = {
                'AvailabilityZone': a_zone
            }
            try:
                response = res = client.run_instances(**arguments_dict)
                if res and len(res['Instances']) >= 1:
                    break
            except ClientError as e:
                print(f"Failed to launch in AZ {a_zone} with Error: {e}")
                continue
    else:
        response = client.run_instances(**arguments_dict)

    if not response or len(response["Instances"]) < 1:
        raise Exception(
            "Unable to launch the instance. \
                         Did not return any response"
        )

    return response["Instances"][0]


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
    user = "ubuntu" if instance["ImageId"] in [UBUNTU_16_BASE_DLAMI] else "ec2-user"
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


def execute_ec2_training_test(connection, ecr_uri, test_cmd, region=DEFAULT_REGION, executable="bash"):
    if executable not in ("bash", "python"):
        raise RuntimeError(f"This function only supports executing bash or python commands on containers")
    if executable == "bash":
        executable = os.path.join(os.sep, 'bin', 'bash')
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    # Run training command
    connection.run(
        f"{docker_cmd} run --name ec2_training_container -v {container_test_local_dir}:{os.path.join(os.sep, 'test')}"
        f" -itd {ecr_uri}",
        hide=True,
    )
    return connection.run(
        f"{docker_cmd} exec --user root ec2_training_container {executable} -c '{test_cmd}'",
        hide=True,
        timeout=3000,
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


def execute_ec2_training_performance_test(connection, ecr_uri, test_cmd, region=DEFAULT_REGION):
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    connection.run(f"{docker_cmd} pull -q {ecr_uri}")

    # Run training command, display benchmark results to console
    connection.run(
        f"{docker_cmd} run --user root -e COMMIT_INFO={os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')} -v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {ecr_uri} "
        f"{os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}"
    )


def execute_ec2_inference_performance_test(connection, ecr_uri, test_cmd, region=DEFAULT_REGION):
    docker_cmd = "nvidia-docker" if "gpu" in ecr_uri else "docker"
    container_test_local_dir = os.path.join("$HOME", "container_tests")

    # Make sure we are logged into ECR so we can pull the image
    connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    connection.run(f"{docker_cmd} pull -q {ecr_uri}")

    # Run training command, display benchmark results to console
    repo_name, image_tag = ecr_uri.split("/")[-1].split(":")
    container_name = f"{repo_name}-performance-{image_tag}-ec2"
    connection.run(
        f"{docker_cmd} run -d --name {container_name} "
        f"-e COMMIT_INFO={os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')} "
        f"-v {container_test_local_dir}:{os.path.join(os.sep, 'test')} {ecr_uri}"
    )
    try:
        connection.run(f"{docker_cmd} exec --user root {container_name} " f"{os.path.join(os.sep, 'bin', 'bash')} -c {test_cmd}")
    except Exception as e:
        raise Exception("Failed to exec benchmark command.\n", e)
    finally:
        connection.run(f"docker rm -f {container_name}")
