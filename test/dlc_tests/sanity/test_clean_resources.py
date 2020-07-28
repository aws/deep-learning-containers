import datetime

from dateutil.tz import tzlocal

import pytest
import boto3

from invoke.context import Context

from test.test_utils import LOGGER
from test.test_utils.eks import eks_setup


def delete_idle_eks_clusters(max_time=240):
    client = boto3.client("eks")
    deleted_clusters = []

    ctx = Context()

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
            ctx.run(f"eksctl delete cluster --name {cluster_name}")
            deleted_clusters.append(cluster_name)

    return deleted_clusters


@pytest.mark.model("N/A")
def test_cleanup_eks_resouces():
    # Setup eks requirements
    eks_setup()
    deleted_clusters = delete_idle_eks_clusters()
    LOGGER.info(f"Deleted EKS clusters: {deleted_clusters}")
