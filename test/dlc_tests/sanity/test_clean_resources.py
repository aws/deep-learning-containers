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

    clusters_to_delete = {}

    all_clusters  = []
    while next_token:
        if next_token == "first":
            response = client.list_clusters(maxResults=100)
        else:
            response = client.list_clusters(maxResults=100, nextToken=next_token)
        all_clusters += response.get('clusters')
        next_token = response.get('nextToken')

    for cluster_name in all_clusters:
        cluster = client.describe_cluster(name=cluster_name)
        create_time = cluster.get('cluster').get('createdAt')
        now_time = datetime.datetime.now(tzlocal())
        time_delta = now_time - create_time
        if time_delta.seconds / 60 > max_time:
            # instantiate a cluster to delete
            clusters_to_delete[cluster_name] = {}
            # Check cfn stacks
            cfn_resp = cfn_client.list_stacks()
            summaries = cfn_resp.get("StackSummaries")

            for stack in summaries:
                stack_name = stack.get("StackName")
                if "eksctl" in stack_name and cluster_name in stack_name:
                    if "nodegroup" in stack_name:
                        clusters_to_delete[cluster_name]['nodegroup'] = stack
                    else:
                        clusters_to_delete[cluster_name]['cluster'] = stack

    # Delete nodegroups
    deleted_nodegroups = delete_cfn_stacks(clusters_to_delete, cfn_client, 'nodegroup')

    # Wait for nodegroups to be deleted
    for nodegroup in deleted_nodegroups:
        cfn_waiter.wait(StackName=nodegroup)

    # Delete clusters
    deleted_clusters = delete_cfn_stacks(clusters_to_delete, cfn_client, 'cluster')

    # Wait for clusters to be deleted
    for eks_cluster in deleted_clusters:
        cfn_waiter.wait(StackName=eks_cluster)

    for eks_cluster_name, _ in clusters_to_delete.items():
        try:
            client.delete_cluster(name=eks_cluster_name)
        except ClientError:
            LOGGER.info(f"Cluster {cluster_name} already deleted.")

    return clusters_to_delete


def delete_cfn_stacks(clusters_to_delete, client, stack_type):
    deleted_stack_names = []
    for c_name, stacks in clusters_to_delete.items():
        stack = stacks.get(stack_type)
        if stack:
            stack_name = stack.get('StackName')
            if stack.get("StackStatus") != "DELETE_IN_PROGRESS":
                client.delete_stack(StackName=stack_name)
                LOGGER.info(f"Deleting stack {stack_name}")
            deleted_stack_names.append(stack_name)
    return deleted_stack_names


@pytest.mark.model("N/A")
@pytest.mark.skipif('mxnet' not in os.getenv("TEST_TRIGGER"), reason='only run once')
def test_cleanup_eks_resouces():
    # Setup eks requirements
    deleted_clusters = delete_idle_eks_clusters()
    LOGGER.info(f"Deleted EKS clusters: {deleted_clusters}")
