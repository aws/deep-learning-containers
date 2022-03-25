from kubernetes import client, config
from datetime import datetime
from invoke import run
import pytz
import boto3

JOB_TIMEOUT = 3
EKS_CLUSTERS = ["mxnet1-PR"]
# EKS_CLUSTERS = ["mxnet-PR", "pytorch-PR", "tensorflow-PR", "mxnet-MAINLINE", "pytorch-MAINLINE", "tensorflow-MAINLINE"]
EKS_CLUSTER_MANAGER_ROLE_NAME = "clusterManagerRole"


def get_run_time(creation_time):
    current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
    diff = current_time - creation_time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600

    return hours


def delete_resources(list_item, k8s_api, job_type, namespace):

    for item in list_item.items:
        item_name = item.metadata.name
        item_creation_time = item.metadata.creation_timestamp
        print(f"Resource name {item_name}")
        print(f"Resource creation time {item_creation_time}")
        # Do not delete the kubeflow mxnet operator
        if "mxnet-operator" in item_name:
            continue

        hours = get_run_time(item_creation_time)
        print(f"Resource up time {hours}")
        if hours >= JOB_TIMEOUT:
            print(f"Deleting resource {item_name}")
            if job_type == "deployment":
                k8s_api.delete_namespaced_deployment(item_name, namespace)
            if job_type == "pod":
                k8s_api.delete_namespaced_pod(item_name, namespace)


def run_cron_job():
    core_v1_api = client.CoreV1Api()
    apps_v1_api = client.AppsV1Api()

    list_deployment_default = apps_v1_api.list_namespaced_deployment(namespace="default")
    list_pod_default = core_v1_api.list_namespaced_pod(namespace="default")

    delete_resources(list_deployment_default, apps_v1_api, "deployment", "default")
    delete_resources(list_pod_default, core_v1_api, "pod", "default")


def main():

    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity().get("Account")
    EKS_CLUSTER_MANAGER_ROLE = f"arn:aws:iam::{account_id}:role/{EKS_CLUSTER_MANAGER_ROLE_NAME}"

    for cluster in EKS_CLUSTERS:
        run(
            f"eksctl utils write-kubeconfig --cluster {cluster} --authenticator-role-arn {EKS_CLUSTER_MANAGER_ROLE} --region us-west-2"
        )
        config.load_kube_config()
        _, active_context = config.list_kube_config_contexts()
        print(f"EKS Cluster {active_context['name']}")
        run_cron_job()


main()
