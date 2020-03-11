import boto3
import pytest

import test.dlc_tests.ecs.utils as ecs_utils


def test_dummy(mxnet_inference):
    print(mxnet_inference)

def test_ecs_create_and_delete_cluster():
    cluster_name = 'saimidu-cicd-test-cluster'
    cluster_arn = ecs_utils.create_ecs_cluster(cluster_name, region='us-west-2')
    ecs_utils.check_ecs_cluster_status(cluster_arn, 'ACTIVE', region='us-west-2')
    ecs_utils.delete_ecs_cluster(cluster_arn, region='us-west-2')
