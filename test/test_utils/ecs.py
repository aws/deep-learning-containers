"""
Helper functions for ECS Integration Tests
"""
from retrying import retry
import boto3

from test.test_utils import DEFAULT_REGION
import test.test_utils.ec2 as ec2_utils
from test.test_utils import get_mms_run_command

ECS_AMI_ID = {"cpu": "ami-0fb71e703258ab7eb", "gpu": "ami-0a36be2e955646bb2"}

ECS_TENSORFLOW_INFERENCE_PORT_MAPPINGS = [
    {"containerPort": 8500, "hostPort": 8500, "protocol": "tcp"},
    {"containerPort": 8501, "hostPort": 8501, "protocol": "tcp"},
    {"containerPort": 80, "protocol": "tcp"},
]

ECS_MXNET_PYTORCH_INFERENCE_PORT_MAPPINGS = [
    {"containerPort": 8081, "hostPort": 8081, "protocol": "tcp"},
    {"containerPort": 8080, "hostPort": 80, "protocol": "tcp"},
]

ECS_INSTANCE_ROLE_ARN = "arn:aws:iam::669063966089:instance-profile/ecsInstanceRole"
ECS_S3_TEST_BUCKET = "s3://dlcinfra-ecs-testscripts/container_tests"
TENSORFLOW_MODELS_BUCKET = "s3://tensoflow-trained-models"


def retry_if_value_error(exception):
    """
    Helper to let retry know whether to re-run
    :param exception: Type of exception received
    :return: <bool> True if test failed with ValueError
    """
    return isinstance(exception, ValueError)


def get_tensorflow_model_name(processor, model_name):
    """
    Helper function to get tensorflow model name
    :param processor: Processor Type
    :param model_name: Name of model to be used
    :return: File name for model being used
    """
    tensorflow_models = {
        "saved_model_half_plus_two": {
            "cpu": "saved_model_half_plus_two_cpu",
            "gpu": "saved_model_half_plus_two_gpu",
            "eia": "saved_model_half_plus_two",
        },
        "saved_model_half_plus_three": {"eia": "saved_model_half_plus_three"},
    }
    if model_name in tensorflow_models:
        return tensorflow_models[model_name][processor]
    else:
        raise Exception(f"No entry found for model {model_name} in dictionary")


@retry(
    stop_max_attempt_number=12,
    wait_fixed=10000,
    retry_on_exception=retry_if_value_error,
)
def check_ecs_cluster_status(cluster_arn_or_name, status, region=DEFAULT_REGION):
    """
    Compares the cluster state (Health Checks).
    Retries 12 times with 10 seconds gap between retries
    :param cluster_arn_or_name: Cluster ARN or Cluster Name
    :param status: Expected status
    :param region: Region where cluster is located
    :return: <bool> True if cluster status matches expected status, else Exception
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        response = ecs_client.describe_clusters(clusters=[cluster_arn_or_name])
        if response["failures"]:
            raise Exception(
                f"Failures in describe cluster. Error - Expected {status} but got {response['failures']}"
            )
        elif (
            response["clusters"][0]["clusterArn"] == cluster_arn_or_name
            and response["clusters"][0]["status"] == status
        ):
            return True
        else:
            raise ValueError(f"Cluster status is not {status}")
    except Exception as e:
        raise Exception(e)


def create_ecs_cluster(cluster_name, region=DEFAULT_REGION):
    """
    Create an ecs cluster
    :param cluster_name: Cluster Name
    :param region: Region where cluster is located
    :return: Cluster ARN if cluster created and is in ACTIVE state else throw Exception
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        response = ecs_client.create_cluster(clusterName=cluster_name)
        cluster_arn = response["cluster"]["clusterArn"]
        return cluster_arn
    except Exception as e:
        raise Exception(f"Failed to launch cluster - {e}")


def list_ecs_container_instances(cluster_arn_or_name, filter_value=None, status=None, region=DEFAULT_REGION):
    """
    List container instances in a cluster.
    :param cluster_arn_or_name: Cluster ARN or Cluster Name
    :param filter_value:
    :param status:
    :param region: Region where cluster is located
    :return: <list> List of container instances
    """
    ecs_client = boto3.Session(region_name=region).client("ecs")
    try:
        arguments_dict = {"cluster": cluster_arn_or_name}
        if filter_value:
            arguments_dict["filter"] = filter_value
        if status:
            arguments_dict["status"] = status
            ecs_client = boto3.Session(region_name=region).client("ecs")
        response = ecs_client.list_container_instances(**arguments_dict)
        return response["containerInstanceArns"]
    except Exception as e:
        raise Exception(f"Failed list instances with given arguments. Exception - {e}")


def attach_ecs_worker_node(worker_instance_type, ami_id, cluster_name, cluster_arn=None, region=DEFAULT_REGION):
    """
    Launch a worker instance in a cluster.
    :param worker_instance_type:
    :param ami_id:
    :param cluster_name:
    :param cluster_arn:
    :param region:
    :return: <tuple> instance_id, public_ip_address
    """
    ecs_user_data = f"""#!/bin/bash
    echo ECS_CLUSTER={cluster_name} >> /etc/ecs/ecs.config"""

    instc = ec2_utils.launch_instance(
        ami_id,
        region=region,
        instance_type=worker_instance_type,
        user_data=ecs_user_data,
        iam_instance_profile_arn=ECS_INSTANCE_ROLE_ARN,
        instance_name=f"ecs worker {cluster_name}",
    )

    instance_id = instc["InstanceId"]
    public_ip_address = ec2_utils.get_public_ip(instance_id, region=region)
    ec2_utils.check_instance_state(instance_id, state="running", region=region)
    ec2_utils.check_system_state(
        instance_id, system_status="ok", instance_status="ok", region=region
    )

    list_container_filter = (
        f"ec2InstanceId in ['{instance_id}'] and agentConnected==true"
    )
    if cluster_arn is None:
        cluster_arn = cluster_name
    container_arns = list_ecs_container_instances(
        cluster_arn, list_container_filter, "ACTIVE", region
    )

    if not container_arns:
        raise Exception(
            f"No ACTIVE container instance found on instance-id {instance_id} in cluster {cluster_arn}"
        )
    return instance_id, public_ip_address


def register_ecs_task_definition(
    family_name,
    image,
    log_group_name,
    log_stream_prefix,
    num_cpu,
    memory,
    entrypoint=None,
    container_command=None,
    health_check=None,
    inference_accelerators=None,
    port_mappings=None,
    environment=None,
    num_gpu=None,
    region=DEFAULT_REGION,
):
    """
    Register a task definition
    :param family_name: Cluster Name
    :param image: ECR URI for docker image to be used
    :param log_group_name:
    :param log_stream_prefix:
    :param num_cpu:
    :param memory:
    :param entrypoint: Entrypoint to be run by ECS Task
    :param container_command: Container command to be executed
    :param health_check: Health check command that can be executed
    :param inference_accelerators: EI accelerator type
    :param port_mappings:
    :param environment:
    :param num_gpu:
    :param region:
    :return: <tuple> task_family, task_revision
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        arguments_dict = {
            "family": family_name,
            "networkMode": "bridge",
            "containerDefinitions": [
                {
                    "name": family_name,
                    "image": image,
                    "cpu": num_cpu,
                    "memory": memory,
                    "essential": True,
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": log_group_name,
                            "awslogs-region": region,
                            "awslogs-stream-prefix": log_stream_prefix,
                            "awslogs-create-group": "true",
                        },
                    },
                },
            ],
            "volumes": [],
            "placementConstraints": [],
            "requiresCompatibilities": ["EC2"],
        }
        if port_mappings:
            arguments_dict["containerDefinitions"][0]["portMappings"] = port_mappings
        if environment:
            arguments_dict["containerDefinitions"][0]["environment"] = environment
        if entrypoint:
            arguments_dict["containerDefinitions"][0]["entryPoint"] = entrypoint
        if container_command:
            arguments_dict["containerDefinitions"][0]["command"] = container_command
        if health_check:
            arguments_dict["containerDefinitions"][0]["healthCheck"] = health_check
        if inference_accelerators:
            arguments_dict["containerDefinitions"][0]["resourceRequirements"] = [
                {
                    "type": "InferenceAccelerator",
                    "value": inference_accelerators["deviceName"],
                }
            ]
            arguments_dict["inferenceAccelerators"] = [
                {
                    "deviceName": inference_accelerators["deviceName"],
                    "deviceType": inference_accelerators["deviceType"],
                }
            ]
        if num_gpu:
            if not isinstance(num_gpu, str):
                raise Exception(
                    f"Invalid type for argument num_gpu, type: {num_gpu}. valid type: <string>"
                )
            else:
                arguments_dict["containerDefinitions"][0]["resourceRequirements"] = [
                    {"value": num_gpu, "type": "GPU"},
                ]
        response = ecs_client.register_task_definition(**arguments_dict)
        return (
            response["taskDefinition"]["family"],
            response["taskDefinition"]["revision"],
        )
    except Exception as e:
        raise Exception(
            f"Failed to register task definition {family_name}. Exception - {e}"
        )


def create_ecs_service(cluster_name, service_name, task_definition, region=DEFAULT_REGION):
    """
    Create an ECS service with EC2 launch type and REPLICA scheduling strategy.
    Wait till the service gets into RUNNING state
    :param cluster_name:
    :param service_name:
    :param task_definition:
    :param region:
    :return: service name
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        # Create Service
        response = ecs_client.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=task_definition,
            desiredCount=1,
            launchType="EC2",
            schedulingStrategy="REPLICA",
        )
        # Wait for the service to get into ACTIVE state
        waiter = ecs_client.get_waiter("services_stable")
        waiter.wait(cluster=cluster_name, services=[response["service"]["serviceName"]])
        return response["service"]["serviceName"]
    except Exception as e:
        raise Exception(
            f"Failed to create service: {service_name} with task definition: {task_definition}. "
            f"Exception - {e}"
        )


@retry(stop_max_attempt_number=15, wait_fixed=20000)
def check_running_task_for_ecs_service(cluster_arn_or_name, service_name, region=DEFAULT_REGION):
    """
    Check for running tasks in the service.
    Retries 15 times with 20 seconds gap between retries
    :param cluster_arn_or_name:
    :param service_name:
    :param region:
    :return: True if service has RUNNING tasks else throws Exception
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        response = ecs_client.list_tasks(
            cluster=cluster_arn_or_name,
            serviceName=service_name,
            desiredStatus="RUNNING",
        )
        task_arns = response["taskArns"]

        if not task_arns:
            raise Exception(
                f"Failed to find task with RUNNING status in {service_name} service"
            )
        else:
            return True

    except Exception as e:
        raise Exception(f"Failed to list task. Exception - {e}")


def update_ecs_service(cluster_arn_or_name, service_name, desired_count, region=DEFAULT_REGION):
    """
    Update desired count of tasks in a service
    :param cluster_arn_or_name:
    :param service_name:
    :param desired_count:
    :param region:
    :return: Exception if API call fails
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        ecs_client.update_service(
            cluster=cluster_arn_or_name,
            service=service_name,
            desiredCount=desired_count,
        )
    except Exception as e:
        raise Exception(
            f"Failed to update desired count for service {service_name} to {desired_count}. Exception {e}"
        )


def create_ecs_task(cluster_arn_or_name, task_definition, region=DEFAULT_REGION):
    """
    Create an ECS task with EC2 launch type in given cluster.
    Wait till the task gets into RUNNING state
    :param cluster_arn_or_name:
    :param task_definition:
    :param region:
    :return: task_arn if task gets into RUNNING state
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")

        response = ecs_client.run_task(
            cluster=cluster_arn_or_name,
            taskDefinition=task_definition,
            count=1,
            launchType="EC2",
        )

        if response["failures"]:
            raise Exception(f"Failures in create task - {response['failures']}")
        elif ecs_task_waiter(
            cluster_arn_or_name,
            [response["tasks"][0]["taskArn"]],
            "tasks_running",
            waiter_delay=6,
        ):
            return response["tasks"][0]["taskArn"]
    except Exception as e:
        raise Exception(
            f"Failed to create task with task definition {task_definition}. Reason - {e}"
        )


def ecs_task_waiter(
        cluster_arn_or_name, task_arns, status, waiter_delay=30, waiter_max_attempts=100, region=DEFAULT_REGION,
):
    """
    Waiter for ECS tasks to get into status defined by "status" parameter.
    Retries "waiter_max_attempts" times with "waiter_delay" seconds gap between retries
    :param cluster_arn_or_name:
    :param task_arns:
    :param status:
    :param waiter_delay:
    :param waiter_max_attempts:
    :param region:
    :return: True or Exception if status is not met in given time
    """

    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        waiter = ecs_client.get_waiter(status)
        waiter.wait(
            cluster=cluster_arn_or_name,
            tasks=task_arns,
            WaiterConfig={"Delay": waiter_delay, "MaxAttempts": waiter_max_attempts},
        )
        return True
    except Exception as e:
        raise Exception(f"Tasks {task_arns} not in {status} state. Exception - {e}")


def describe_ecs_task_exit_status(cluster_arn_or_name, task_arn, region=DEFAULT_REGION):
    """
    Describes a specified task and checks for the exit code returned from the containers in the task is equal to zero
    :param cluster_arn_or_name:
    :param task_arn:
    :param region:
    :return: True if exit code is zero, else False
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        response = ecs_client.describe_tasks(cluster=cluster_arn_or_name, tasks=[task_arn])
        return_code = []
        if response["failures"]:
            raise Exception(f"Failures in describe tasks - {response['failures']}")
        else:
            for container in response["tasks"][0]["containers"]:
                return_code.append(container["exitCode"] == 0)
            return all(return_code)
    except Exception as e:
        raise Exception(
            f"Failed to describe task {task_arn} in cluster {cluster_arn_or_name}. Exception - {e}"
        )


def stop_ecs_task(cluster_arn_or_name, task_arn, region=DEFAULT_REGION):
    """
    Stops a running task
    :param cluster_arn_or_name:
    :param task_arn:
    :param region:
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        ecs_client.stop_task(cluster=cluster_arn_or_name, task=task_arn)
    except Exception as e:
        raise Exception(
            f"Failed to stop task {task_arn} in cluster {cluster_arn_or_name}. Exception - {e}"
        )


def delete_ecs_service(cluster_arn_or_name, service_name, region=DEFAULT_REGION):
    """
    Delete a service
    :param cluster_arn_or_name:
    :param service_name:
    :param region:
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        ecs_client.delete_service(
            cluster=cluster_arn_or_name, service=service_name, force=True
        )
    except Exception as e:
        raise Exception(f"Failed to delete service {service_name}. Exception {e}")


def deregister_ecs_task_definition(task_family, revision, region=DEFAULT_REGION):
    """
    De-register a task definition
    :param task_family:
    :param revision:
    :param region:
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        ecs_client.deregister_task_definition(
            taskDefinition=f"{task_family}:{revision}"
        )
    except Exception as e:
        raise Exception(
            f"Failed to deregister task definition {task_family}:{revision}. Reason - {e}"
        )


def deregister_ecs_container_instances(cluster_arn_or_name, container_instances, region=DEFAULT_REGION):
    """
    De-register all container instances in a cluster
    :param cluster_arn_or_name:
    :param container_instances:
    :param region:
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        for container_instance in container_instances:
            ecs_client.deregister_container_instance(
                cluster=cluster_arn_or_name,
                containerInstance=container_instance,
                force=True,
            )
    except Exception as e:
        raise Exception(f"Failed to delete cluster. Exception - {e}")


def delete_ecs_cluster(cluster_arn, region=DEFAULT_REGION):
    """
    Delete ECS cluster.
    Deregister all container instances from this cluster before deleting it (This is must).
    :param cluster_arn:
    :param region:
    :return: True if cluster INACTIVE after deletion else Exception
    """
    try:
        ecs_client = boto3.Session(region_name=region).client("ecs")
        # List container instances
        container_instances = list_ecs_container_instances(cluster_arn, region=region)
        # Deregister all container instances from this cluster
        deregister_ecs_container_instances(cluster_arn, container_instances, region)
        # Delete Cluster
        ecs_client.delete_cluster(cluster=cluster_arn)
        if check_ecs_cluster_status(cluster_arn, "INACTIVE"):
            return True
        else:
            raise Exception("Cluster health check failed")
    except Exception as e:
        raise Exception(f"Failed to delete cluster. Exception - {e}")


def tear_down_ecs_inference_service(cluster_arn, service_name, task_family, revision):
    """
    Function to clean up ECS task definition, service resources if exist
    :param cluster_arn:
    :param service_name:
    :param task_family:
    :param revision:
    """

    if task_family and revision:
        deregister_ecs_task_definition(task_family, revision)
    else:
        print("Skipped - De-register task definition")

    if service_name and cluster_arn:
        # Scale down desired count to 0
        update_ecs_service(cluster_arn, service_name, 0)
        delete_ecs_service(cluster_arn, service_name)
    else:
        print("Skipped - Service and cluster deletion")


def tear_down_ecs_training_task(cluster_arn, task_arn, task_family, revision):
    """
    Function to clean up ECS training task resources - task and task definition if exists
    :param cluster_arn:
    :param task_arn:
    :param task_family:
    :param revision:
    """

    if task_family and revision:
        deregister_ecs_task_definition(task_family, revision)
    else:
        print("Skipped - De-register task definition")

    if task_arn and cluster_arn:
        stop_ecs_task(cluster_arn, task_arn)
    else:
        print("Skipped - Stopping task")


def cleanup_worker_node_cluster(worker_instance_id, cluster_arn):
    """
    Function to clean up ECS worker node and cluster
    :param worker_instance_id:
    :param cluster_arn:
    """
    if worker_instance_id:
        ec2_utils.terminate_instance(worker_instance_id)
    else:
        print("Skipped - terminating ecs worker node")
    if cluster_arn:
        delete_ecs_cluster(cluster_arn)
    else:
        print("Skipped - deleting cluster")


def get_ecs_port_mappings(framework):
    """
    Get method for ECS inference port mapping parameter
    :param framework:
    :return: <list> JSON containing the port mappings for ECS inference
        Exception if framework port mappings not available
    """
    if framework == "tensorflow":
        return ECS_TENSORFLOW_INFERENCE_PORT_MAPPINGS
    elif framework in ["mxnet", "pytorch"]:
        return ECS_MXNET_PYTORCH_INFERENCE_PORT_MAPPINGS
    else:
        raise Exception("Framework not Implemented")


def get_ecs_tensorflow_environment_variables(processor, model_name):
    """
    Get method for environment variables for tensorflow inference via S3 on ECS
    Requirement: Model should be hosted in S3 location defined in TENSORFLOW_MODELS_PATH
    :param processor:
    :param model_name:
    :return: <list> JSON
    """
    model_name = get_tensorflow_model_name(processor, model_name)
    ecs_tensorflow_inference_environment = [
        {"name": "MODEL_NAME", "value": model_name},
        {"name": "MODEL_BASE_PATH", "value": TENSORFLOW_MODELS_BUCKET},
    ]

    return ecs_tensorflow_inference_environment


def build_ecs_training_command(test_string):
    """
    Construct the command to send to the container for running scripts in ECS automation
    :param test_string:
    :return: <list> command to send to the container
    """
    return [
        f"pip install -U awscli && mkdir -p /test/logs && aws s3 cp {ECS_S3_TEST_BUCKET} /test/ --recursive "
        f"&& chmod +x -R /test/ && {test_string}"
    ]


# TODO: Move this to testspec.yml for ecs tests
"""
def upload_tests_for_ecs(datetime_suffix):
    aws_access_key_id, aws_secret_access_key = utils.get_private_key(test_helper.AMI_MATERIALSET)
    with hide('running'):
        run("aws configure set aws_access_key_id {}".format(aws_access_key_id))
        run("aws configure set aws_secret_access_key {}".format(aws_secret_access_key))
        run("aws configure set region {}".format(cfg.iad_region))
    global ECS_S3_TEST_BUCKET
    ECS_S3_TEST_BUCKET = ECS_S3_TEST_BUCKET.format(datetime_suffix)
    run("aws s3 rm {}/ --recursive".format(ECS_S3_TEST_BUCKET), warn_only=True)
    run("aws s3 cp {}/container_tests/ {} --recursive".format(test_helper.SRC_DIR, ECS_S3_TEST_BUCKET,
                                                              datetime_suffix))
"""


def ecs_inference_test_executor(
    docker_image_uri,
    framework,
    job,
    processor,
    cluster_name,
    cluster_arn,
    datetime_suffix,
    model_name,
    num_cpus,
    memory,
    num_gpus,
    test_args,
):
    """
    Create a service in an existing cluster, run the test using the arguments passed in
    *test_args and scales down and deletes the service
    This is a helper function to run help run ECS inference tests. Cluster can be reused to run N
    number of tests but each model will need a new service
    :param docker_image_uri:
    :param framework:
    :param job:
    :param processor:
    :param cluster_name:
    :param cluster_arn:
    :param datetime_suffix:
    :param model_name:
    :param num_cpus:
    :param memory:
    :param num_gpus:
    :param test_args:
    :return: <bool> True if test passed else False
    """
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = setup_ecs_inference_service(
            docker_image_uri,
            framework,
            job,
            processor,
            cluster_name,
            cluster_arn,
            datetime_suffix,
            model_name,
            num_cpus,
            memory,
            num_gpus,
        )
        if service_name is None:
            return [False]
        return_codes = []
        for args in test_args:
            test_function, test_function_arguments = args[0], args[1:]
            return_codes.append(test_function(*test_function_arguments))
        return return_codes
    finally:
        tear_down_ecs_inference_service(
            cluster_arn, service_name, task_family, revision
        )


def setup_ecs_inference_service(
    docker_image_uri,
    framework,
    job,
    processor,
    cluster_name,
    cluster_arn,
    datetime_suffix,
    model_name,
    num_cpus,
    memory,
    num_gpus,
):
    """
    Function to setup Inference service on ECS
    :param docker_image_uri:
    :param framework:
    :param job:
    :param processor:
    :param cluster_name:
    :param cluster_arn:
    :param datetime_suffix:
    :param model_name:
    :param num_cpus:
    :param memory:
    :param num_gpus:
    :return: <tuple> service_name, task_family, revision if all steps passed else Exception
        Cleans up the resources if any step fails
    """
    service_name = task_family = revision = None
    try:
        port_mappings = get_ecs_port_mappings(framework)
        log_group_name = "/ecs/{}-{}-{}".format(framework, job, processor)
        # Below values here are just for sanity
        arguments_dict = {
            "family_name": cluster_name,
            "image": docker_image_uri,
            "log_group_name": log_group_name,
            "log_stream_prefix": datetime_suffix,
            "port_mappings": port_mappings,
            "num_cpu": num_cpus,
            "memory": memory,
        }

        if processor == "gpu" and num_gpus:
            arguments_dict["num_gpu"] = num_gpus
        if processor == "tensorflow":
            arguments_dict["environment"] = get_ecs_tensorflow_environment_variables(
                processor, model_name
            )
        elif framework in ["mxnet", "pytorch"]:
            arguments_dict["container_command"] = [
                get_mms_run_command(model_name, processor)
            ]

        task_family, revision = register_ecs_task_definition(**arguments_dict)
        print(f"Created Task definition - {task_family}:{revision}")

        service_name = create_ecs_service(
            cluster_name, f"service-{cluster_name}", f"{task_family}:{revision}"
        )
        print(
            f"Created ECS service - {service_name} with cloudwatch log group - {log_group_name} "
            f"log stream prefix - {datetime_suffix}/{cluster_name}"
        )
        if check_running_task_for_ecs_service(cluster_name, service_name):
            print("Service status verified as running. Running inference ...")
        else:
            raise Exception(f"No task running in the service: {service_name}")
        return service_name, task_family, revision
    except Exception as e:
        print(f"Setup failure Exception - {e}")
        tear_down_ecs_inference_service(
            cluster_arn, service_name, task_family, revision
        )
    return None, None, None
