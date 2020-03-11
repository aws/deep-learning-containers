import boto3
from retrying import retry


def launch_instance(ami_id, instance_type, region='us-west-2', user_data=None, iam_instance_profile_arn=None,
                    instance_name=''):
    """Launch an instance
    """
    if not ami_id:
        raise Exception("No ami_id provided")
    client = boto3.Session(region_name=region).client('ec2')

    # Construct the dictionary with the arguments for API call
    arguments_dict = {
        "ImageId": ami_id,
        "InstanceType": instance_type,
        "MaxCount": 1,
        "MinCount": 1,
        "TagSpecifications": [
            {
                'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': 'Name',
                        'Value': f'CI-CD-instance {instance_name}'
                    }
                ]
            },
        ]
    }
    if user_data:
        arguments_dict["UserData"] = user_data
    if iam_instance_profile_arn:
        arguments_dict["IamInstanceProfile"] = {
            "Arn": iam_instance_profile_arn
        }
    response = client.run_instances(**arguments_dict)

    if not response or len(response['Instances']) < 1:
        raise Exception("Unable to launch the instance. \
                         Did not return any response")

    return response['Instances'][0]


def get_instance_from_id(instance_id, region='us-west-2'):
    """get instance id from instance
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client('ec2')
    instance = client.Instance(instance_id)
    if not instance:
        raise Exception("Unable to launch the instance. \
                         Did not return any reservations object")
    return instance


@retry(stop_max_attempt_number=8, wait_fixed=120000)
def get_public_ip(instance_id, region='us-west-2'):
    instance = get_instance_from_id(instance_id, region)
    if not instance.public_ip_address:
        raise Exception("IP address not yet available")
    return instance.public_ip_address


def get_instance_state(instance_id, region='us-west-2'):
    instance = get_instance_from_id(instance_id, region)
    return instance.state['Name']


@retry(stop_max_attempt_number=8, wait_fixed=120000)
def check_instance_state(instance_id, state="running", region='us-west-2'):
    """Compares the instance state with the state argument.
       Retries 8 times with 120 seconds gap between retries
    """
    instance_state = get_instance_state(instance_id, region)
    if state != instance_state:
        raise Exception(f"Instance {instance_id} not in {state} state")
    return instance_state


def get_system_state(instance_id, region='us-west-2'):
    """Returns health checks state for instances
    """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client('ec2')
    response = client.describe_instance_status(InstanceIds=[instance_id])
    if not response:
        raise Exception("Unable to launch the instance. \
                         Did not return any reservations object")
    instance_status_list = response['InstanceStatuses']
    if not instance_status_list:
        raise Exception("Unable to launch the instance. \
                         Did not return any reservations object")
    if len(instance_status_list) < 1:
        raise Exception("The instance id seems to be incorrect {}. \
                         reservations seems to be empty".format(instance_id))

    instance_status = instance_status_list[0]
    return (instance_status['SystemStatus']['Status'],
            instance_status['InstanceStatus']['Status'])


@retry(stop_max_attempt_number=96, wait_fixed=10000)
def check_system_state(instance_id,
                       system_status="ok",
                       instance_status="ok",
                       region="us-west-2"):
    """Compares the system state (Health Checks).
       Retries 96 times with 10 seconds gap between retries
    """
    instance_state = get_system_state(instance_id, region=region)
    if system_status != instance_state[0] or instance_status != instance_state[1]:
        raise Exception("Instance {} not in \
                         required state".format(instance_id))
    return instance_state


def terminate_instance(instance_id, region='us-west-2'):
    """Terminate instances
        """
    if not instance_id:
        raise Exception("No instance id provided")
    client = boto3.Session(region_name=region).client('ec2')
    response = client.terminate_instances(InstanceIds=[instance_id])
    if not response:
        raise Exception("Unable to terminate instance. No response received.")
    instances_terminated = response['TerminatingInstances']
    if not instances_terminated:
        raise Exception("Failed to terminate instance.")
    if instances_terminated[0]['InstanceId'] != instance_id:
        raise Exception("Failed to terminate instance. Unknown error.")
