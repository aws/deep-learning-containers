import datetime
import os

from dateutil.tz import tzlocal

import pytest
import boto3

from botocore.exceptions import ClientError

from test.test_utils import LOGGER


def delete_idle_eks_clusters(max_time=240):
    client = boto3.client("eks")
    cfn_client = boto3.client("cloudformation")
    cfn_waiter = cfn_client.get_waiter('stack_delete_complete')
    deleted_clusters = []
    next_token = "first"
    clusters = []
    while next_token:
        if next_token == "first":
            response = client.list_clusters(maxResults=100)
        else:
            response = client.list_clusters(maxResults=100, nextToken=next_token)
        clusters += response.get('clusters')
        next_token = response.get('nextToken')
    for cluster_name in clusters:
        cluster = client.describe_cluster(name=cluster_name)
        create_time = cluster.get('cluster').get('createdAt')
        now_time = datetime.datetime.now(tzlocal())
        time_delta = now_time - create_time
        if time_delta.seconds / 60 > max_time:
            LOGGER.info(f"deleting cluster {cluster_name} which is older than {max_time / 60} hours old")
            cfn_resp = cfn_client.list_stacks()
            for stack in cfn_resp.get('StackSummaries'):
                stack_name = stack.get("StackName")
                if cluster_name in stack_name and "nodegroup" in stack_name and "eksctl" in stack_name:
                    if stack.get('StackStatus') == "DELETE_COMPLETE":
                        break
                    elif stack.get("StackStatus") != 'DELETE_IN_PROGRESS':
                        cfn_client.delete_stack(StackName=stack_name)
                    cfn_waiter.wait(StackName=stack_name)
                    break
            for cluster_stack in cfn_resp.get('StackSummaries'):
                cluster_stack_name = cluster_stack.get("StackName")
                if cluster_name in cluster_stack_name and "nodegroup" not in cluster_stack_name and "eksctl" in stack_name:
                    if cluster_stack.get("StackStatus") == "DELETE_COMPLETE":
                        break
                    elif stack.get("StackStatus") != 'DELETE_IN_PROGRESS':
                        cfn_client.delete_stack(StackName=cluster_stack_name)
                    cfn_waiter.wait(StackName=cluster_stack_name)
                    break
            try:
                client.delete_cluster(name=cluster_name)
            except ClientError:
                print("Cluster already deleted")
            deleted_clusters.append(cluster_name)
        else:
            LOGGER.info(f"cluster {cluster_name} is less than {max_time / 60} hours old")
    LOGGER.info(f"deleted clusters: {deleted_clusters}")
    return deleted_clusters


@pytest.mark.model("N/A")
@pytest.mark.skipif('mxnet' not in os.getenv("TEST_TRIGGER"), reason='only run once')
def test_cleanup_eks_resouces():
    # Setup eks requirements
    deleted_clusters = delete_idle_eks_clusters()
    LOGGER.info(f"Deleted EKS clusters: {deleted_clusters}")
