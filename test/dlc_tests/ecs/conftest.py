import time

import pytest
import boto3

from test.test_utils import ECS_AML2_GPU_USWEST2


@pytest.fixture(scope="session")
def ecs_client():
    return boto3.client("ecs")


@pytest.fixture(scope="session")
def ecs_cluster_name(request):
    return request.param


@pytest.fixture(scope="session")
def ecs_cluster(request, ecs_client, ecs_cluster_name):
    """
    Fixture to handle spin up and tear down of ecs cluster

    :param request:
    :param ecs_client:
    :return:
    """
    cluster_name = ecs_cluster_name
    ecs_client.create_cluster(
        clusterName=cluster_name
    )

    # Finalizer to delete the ecs cluster
    def delete_ecs_cluster():
        ecs_client.delete_cluster(cluster=cluster_name)

    request.addfinalizer(delete_ecs_cluster)

    # Wait for max 10 minutes for cluster status to be active
    timeout = time.time() + 600
    is_active = False
    while not is_active:
        if time.time() > timeout:
            raise TimeoutError(f"ECS cluster {cluster_name} timed out on creation")
        response = ecs_client.describe_clusters(clusters=[cluster_name])
        if response.get('clusters', [{}])[0].get('status') == 'ACTIVE':
            is_active = True

    return cluster_name


@pytest.fixture(scope="session")
def ecs_ami(request):
    return request.param


@pytest.fixture(scope="session")
def ecs_instance_type(request):
    return request.param


@pytest.mark.timeout(300)
@pytest.fixture(scope="module")
def ecs_container_instance(request, ecs_cluster, ec2_client, ecs_instance_type, ecs_ami):
    """
    Fixture to handle spin up and tear down of ECS container instance

    :param request: pytest request object
    :param ecs_cluster: ecs cluster fixture
    :param ec2_client: boto3 ec2 client
    :param ec2_instance_type: eventually to be used
    :return:
    """
    # Get these from params on the test
    instance_type = ecs_instance_type
    image_id = ecs_ami

    user_data = f"#!/bin/bash\necho ECS_CLUSTER={ecs_cluster} >> /etc/ecs/ecs.config"

    instances = ec2_client.run_instances(
        KeyName="pytest.pem",
        ImageId=image_id,
        InstanceType=instance_type,
        MaxCount=1,
        MinCount=1,
        UserData=user_data,
        IamInstanceProfile={"Name": "ecsInstanceRole"}
    )
    instance_id = instances.get('Instances')[0].get('InstanceId')

    # Define finalizer to terminate instance after this fixture completes
    def terminate_ec2_instance():
        ec2_client.terminate_instances(InstanceIds=[instance_id])

    request.addfinalizer(terminate_ec2_instance)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    return instance_id, ecs_cluster
