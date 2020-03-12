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
def ecs_ami(request):
    return request.param


@pytest.fixture(scope="session")
def ecs_instance_type(request):
    return request.param


@pytest.fixture(scope="session")
def ecs_cluster(request, ecs_client, ecs_cluster_name, ec2_client, ecs_instance_type, ecs_ami):
    """
    Fixture to handle spin up and tear down of ecs cluster
    """
    cluster_name = ecs_cluster_name
    ecs_client.create_cluster(
        clusterName=cluster_name
    )

    # Wait for max 10 minutes for cluster status to be active
    timeout = time.time() + 600
    is_active = False
    while not is_active:
        if time.time() > timeout:
            raise TimeoutError(f"ECS cluster {cluster_name} timed out on creation")
        response = ecs_client.describe_clusters(clusters=[cluster_name])
        if response.get('clusters', [{}])[0].get('status') == 'ACTIVE':
            is_active = True

    # Get these from params on the test
    instance_type = ecs_instance_type
    image_id = ecs_ami

    user_data = f"#!/bin/bash\necho ECS_CLUSTER={cluster_name} >> /etc/ecs/ecs.config"

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
    def delete_cluster():
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        term_waiter = ec2_client.get_waiter('instance_terminated')
        term_waiter.wait(InstanceIds=[instance_id])
        ecs_client.delete_cluster(cluster=cluster_name)

    request.addfinalizer(delete_cluster)

    waiter = ec2_client.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    is_attached = False
    attach_timeout = time.time() + 300
    while not is_attached:
        if time.time() > attach_timeout:
            raise TimeoutError(f"Instance {instance_id} not attached to cluster {cluster_name}")
        response = ecs_client.describe_clusters(cluster=[cluster_name])
        if response.get('clusters', [{}])[0].get('registeredContainerInstancesCount'):
            is_attached = True

    return instance_id, cluster_name
