from os.path import exists, join
import re

import boto3
from invoke import run
from retrying import retry

from test.test_utils import DEFAULT_REGION


EC2_TENSORFLOW_INFERENCE_PORT_MAPPINGS = {
    "gpu": {"py2": "8500", "py3": "8501"},
    "cpu": {"py2": "8502", "py3": "8503"},
    "eia": {"py2": "8504", "py3": "8505"},
}


def launch_instance(
    ami_id,
    instance_type,
    region=DEFAULT_REGION,
    user_data=None,
    iam_instance_profile_arn=None,
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
    client = boto3.Session(region_name=region).client("ec2")

    # Construct the dictionary with the arguments for API call
    arguments_dict = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MaxCount": 1,
        "MinCount": 1,
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "Name", "Value": f"CI-CD {instance_name}"}],
            },
        ],
    }
    if user_data:
        arguments_dict["UserData"] = user_data
    if iam_instance_profile_arn:
        arguments_dict["IamInstanceProfile"] = {"Arn": iam_instance_profile_arn}
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


@retry(stop_max_attempt_number=8, wait_fixed=120000)
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


def get_instance_state(instance_id, region=DEFAULT_REGION):
    """
    Get state of instance using instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <str> State of instance with matching instance ID
    """
    instance = get_instance_from_id(instance_id, region)
    return instance["State"]["Name"]


@retry(stop_max_attempt_number=8, wait_fixed=120000)
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


def get_instance_details(instance_id, region=DEFAULT_REGION):
    """
    Get instance details for instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <dict> Information about instance with matching instance ID
    """
    if not instance_id:
        raise Exception("No instance id provided")
    instance = get_instance_from_id(instance_id, region=DEFAULT_REGION)
    if not instance:
        raise Exception("Could not find instance")
    client = boto3.Session(region_name=region).client("ec2")
    response = client.describe_instance_types(InstanceTypes=[instance["InstanceType"]])
    if not response or not response["InstanceTypes"]:
        raise Exception("Unable to get instance details. No response received.")
    if response["InstanceTypes"][0]["InstanceType"] != instance["InstanceType"]:
        raise Exception(
            f"Bad response received. Requested {instance['InstanceType']} "
            f"but got {response['InstanceTypes'][0]['InstanceType']}"
        )
    return response["InstanceTypes"][0]


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
def get_instance_num_gpus(instance_id, region=DEFAULT_REGION):
    """
    Get total number of GPUs on instance with given instance ID
    :param instance_id: Instance ID to be queried
    :param region: Region where query will be performed
    :return: <int> Number of GPUs on instance with matching instance ID
    """
    instance_info = get_instance_details(instance_id, region=region)
    return sum(gpu_type["Count"] for gpu_type in instance_info["GpuInfo"]["Gpus"])


def host_setup_for_tensorflow_inference(container_name, framework_version):
    home_dir = run("pwd", hide="out").stdout.strip("\n")
    src_location = join(home_dir, f"{container_name}-serving")
    if exists(src_location):
        run(f"rm -rf {src_location}", echo=True)

    if framework_version.startswith('1.'):
        fw_short_version = re.search(r"\d+\.\d+", framework_version).group()
        run(f"git clone -b r{fw_short_version} https://github.com/tensorflow/serving.git {src_location}", echo=True)
    else:
        run(f"mkdir -p {src_location}")
        # tensorflow/serving is not yet updated with scripts for TF 2.1, so using locally modified scripts
        script_location = join("container_tests", "bin", "tensorflow_serving")
        run(f"cp -r {script_location} {src_location}")

    run(
        f"""pip install -qq -U tensorflow=={framework_version}"""
        f""" "tensorflow-serving-api<={framework_version}" """, echo=True
    )

    script = join(src_location, "tensorflow_serving", "example", "mnist_saved_model.py")
    model_path = join(src_location, "models", "mnist")
    run(f"python {script} {model_path}", hide="out")

    return src_location, model_path


def request_tensorflow_inference_grpc(src_location, ip_address="127.0.0.1", port="8500"):
    script = join(src_location, "tensorflow_serving", "example", "mnist_client.py")
    run(f"python {script} --num_tests=1000 --server={ip_address}:{port}", echo=True)


def tensorflow_inference_test_cleanup(src_location):
    run(f"rm -rf {src_location}", warn=True, echo=True)


def get_inference_ec2_test_port_number(image_uri):
    proc = "gpu" if "gpu" in image_uri else "eia" if "eia" in image_uri else "cpu"
    py_ver = "py2" if "py2" in image_uri else "py3"
    if "tensorflow" in image_uri:
        return EC2_TENSORFLOW_INFERENCE_PORT_MAPPINGS.get(proc, {}).get(py_ver, "8500")
    else:
        return "80"
