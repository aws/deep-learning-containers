import datetime
import logging

from dateutil.tz import tzlocal

import boto3

from botocore.exceptions import ClientError, WaiterError


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
HANDLER = logging.StreamHandler()
HANDLER.setLevel(logging.DEBUG)
LOGGER.addHandler(HANDLER)


def delete_idle_eks_clusters(max_time=240):
    client = boto3.client("eks")
    ec2_client = boto3.client("ec2")
    cfn_client = boto3.client("cloudformation")
    cfn_resource = boto3.resource("cloudformation")
    cfn_waiter = cfn_client.get_waiter("stack_delete_complete")

    clusters_to_delete = []
    nodegroup_stacks = []
    cluster_stacks = []

    all_clusters = []

    LOGGER.info("HELLO")

    next_token = "first"
    while next_token:
        if next_token == "first":
            response = client.list_clusters(maxResults=100)
        else:
            response = client.list_clusters(maxResults=100, nextToken=next_token)
        all_clusters += response.get("clusters")
        next_token = response.get("nextToken")

    for cluster_name in all_clusters:
        cluster = client.describe_cluster(name=cluster_name)
        create_time = cluster.get("cluster").get("createdAt")
        now_time = datetime.datetime.now(tzlocal())
        time_delta = now_time - create_time
        if time_delta.total_seconds() / 60 > max_time:
            # instantiate a cluster to delete
            clusters_to_delete.append(cluster_name)

    # Check cfn stacks
    cfn_next = "first"
    summaries = []
    while cfn_next:
        if cfn_next == "first":
            cfn_resp = cfn_client.list_stacks(StackStatusFilter=["CREATE_COMPLETE", "DELETE_IN_PROGRESS"])
        else:
            cfn_resp = cfn_client.list_stacks(
                NextToken=cfn_next, StackStatusFilter=["CREATE_COMPLETE", "DELETE_IN_PROGRESS"]
            )
        summaries += cfn_resp.get("StackSummaries")
        cfn_next = cfn_resp.get("NextToken")

    for stack in summaries:
        stack_name = stack.get("StackName")
        created_at = stack.get("CreationTime")
        now_time = datetime.datetime.now(tzlocal())
        delt = now_time - created_at
        if delt.total_seconds() / 60 > max_time:
            if "eksctl-dlc-" in stack_name:
                if "nodegroup" in stack_name:
                    nodegroup_stacks.append(stack)
                else:
                    cluster_stacks.append(stack)

    # Delete nodegroups
    deleted_nodegroups = delete_cfn_stacks(nodegroup_stacks, cfn_client)

    # Wait for nodegroups to be deleted
    for nodegroup in deleted_nodegroups:
        cfn_waiter.wait(StackName=nodegroup)

    # Delete clusters
    deleted_clusters = delete_cfn_stacks(cluster_stacks, cfn_client)

    # Wait for clusters to be deleted
    for eks_cluster in deleted_clusters:
        try:
            stack_resource_summary = cfn_resource.StackResourceSummary(eks_cluster, "VPC")
            vpc = stack_resource_summary.physical_resource_id
            ec2_client.delete_vpc(VpcID=vpc)
            cfn_waiter.wait(StackName=eks_cluster)
        except WaiterError:
            LOGGER.info(f"Stack failed to delete {eks_cluster}")

    for eks_cluster_name in clusters_to_delete:
        try:
            client.delete_cluster(name=eks_cluster_name)
        except ClientError:
            LOGGER.info(f"Cluster {eks_cluster_name} already deleted.")


def delete_cfn_stacks(stacks_to_delete, client):
    deleted_stack_names = []
    for stack in stacks_to_delete:
        stack_name = stack.get("StackName")
        if stack.get("StackStatus") != "DELETE_IN_PROGRESS":
            client.delete_stack(StackName=stack_name)
            LOGGER.info(f"Deleting stack {stack_name}")
        deleted_stack_names.append(stack_name)
    return deleted_stack_names


if __name__ == "__main__":
    delete_idle_eks_clusters()
