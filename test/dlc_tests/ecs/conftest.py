import time

import pytest
import boto3
import test.test_utils.ecs as ecs_utils


@pytest.fixture(scope="session")
def ecs_client():
    return boto3.client("ecs")


@pytest.fixture(scope="function")
def ecs_cluster_name(request):
    return request.param


@pytest.mark.timeout(300)
@pytest.fixture(scope="function")
def ecs_cluster(request, ecs_client, ecs_cluster_name, region):
    """
    Fixture to handle spin up and tear down of ecs cluster

    :param request:
    :param ecs_client:
    :param ecs_cluster_name:
    :param region:
    :return:
    """
    cluster_name = ecs_cluster_name
    cluster_arn = ecs_utils.create_ecs_cluster(cluster_name, region=region)

    # Finalizer to delete the ecs cluster
    def delete_ecs_cluster():
        ecs_utils.delete_ecs_cluster(cluster_arn, region=region)

    request.addfinalizer(delete_ecs_cluster)

    # Wait for cluster status to be active
    if ecs_utils.check_ecs_cluster_status(cluster_arn, "ACTIVE"):
        return cluster_arn
    raise ecs_utils.ECSClusterCreationException(f'Failed to create ECS cluster - {cluster_name}')


@pytest.fixture(scope="function")
def training_script(request):
    """
    Path that container is expecting training script to be in

    i.e. /test/bin/testTensorFlow
    """
    return request.param


@pytest.fixture(scope="function")
def training_cmd(request, ecs_cluster_name, training_script):
    artifact_folder = f"{ecs_cluster_name}-folder"
    s3_test_artifact_location = ecs_utils.upload_tests_for_ecs(artifact_folder)

    def delete_s3_artifact_copy():
        ecs_utils.delete_uploaded_tests_for_ecs(s3_test_artifact_location)

    request.addfinalizer(delete_s3_artifact_copy)

    return ecs_utils.build_ecs_training_command(s3_test_artifact_location, training_script)


@pytest.fixture(scope="session")
def ecs_ami(request):
    return request.param


@pytest.fixture(scope="session")
def ecs_instance_type(request):
    return request.param


@pytest.mark.timeout(300)
@pytest.fixture(scope="function")
def ecs_container_instance(request, ecs_cluster, ec2_client, ecs_client, ecs_instance_type, ecs_ami):
    """
    Fixture to handle spin up and tear down of ECS container instance

    :param request: pytest request object
    :param ecs_cluster: ecs cluster fixture
    :param ec2_client: boto3 ec2 client
    :param ecs_client: boto3 ecs client
    :param ecs_instance_type: eventually to be used
    :param ecs_ami: eventually to be used
    :return:
    """
    # Get these from params on the test
    instance_type = ecs_instance_type
    image_id = ecs_ami
    cluster_name = ecs_utils.get_ecs_cluster_name(ecs_cluster)

    user_data = f"#!/bin/bash\necho ECS_CLUSTER={ecs_cluster} >> /etc/ecs/ecs.config"

    instances = ec2_client.run_instances(
        KeyName="pytest.pem",
        ImageId=image_id,
        InstanceType=instance_type,
        MaxCount=1,
        MinCount=1,
        UserData=user_data,
        IamInstanceProfile={"Name": "ecsInstanceRole"},
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "Name",
                        "Value": f"CI-CD ecs worker {cluster_name}"
                    }
                ]
            }
        ]
    )
    instance_id = instances.get("Instances")[0].get("InstanceId")

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        terminate_waiter = ec2_client.get_waiter("instance_terminated")
        terminate_waiter.wait(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    is_attached = False

    # Check to see if instance is attached
    while not is_attached:
        # Add sleep to avoid throttling limit
        time.sleep(12)
        response = ecs_client.describe_clusters(clusters=[ecs_cluster])
        if response.get("clusters", [{}])[0].get("registeredContainerInstancesCount"):
            is_attached = True

    return instance_id, ecs_cluster
