from kubernetes import client, config
from datetime import datetime
from invoke import run
import pytz

JOB_TIMEOUT = 3
EKS_CLUSTERS = ["mxnet1-PR"]
#EKS_CLUSTERS = ["mxnet-PR", "pytorch-PR", "tensorflow-PR", "mxnet-MAINLINE", "pytorch-MAINLINE", "tensorflow-MAINLINE"]
EKS_CLUSTER_MANAGER_ROLE = "arn:aws:iam::332057208146:role/clusterManagerRole"


def get_run_time(creation_time):
    current_time = datetime.utcnow().replace(tzinfo=pytz.UTC)
    diff = current_time - creation_time
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    
    return hours

def delete_resources(list_item, k8s_api, job_type, namespace):

    for item in list_item.items:
        item_name = item.metadata.name

        # Do not delete the kubeflow mxnet operator
        if item_name == "mxnet-operator":
            continue
    
        hours = get_run_time(item.metadata.creation_timestamp)

        if hours >= JOB_TIMEOUT:

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




def lambda_handler(event, context):
    # Handle k8s context switch

    for cluster in EKS_CLUSTERS:
        #run(f"aws eks update-kubeconfig --name {cluster} --role-arn {EKS_CLUSTER_MANAGER_ROLE}")
        run(f"eksctl utils write-kubeconfig --cluster {cluster} --authenticator-role-arn {EKS_CLUSTER_MANAGER_ROLE} --region us-west-2")
        config.load_kube_config()
        _, active_context = config.list_kube_config_contexts()
        print(f"EKS Cluster {active_context['name']}")
        #run_cron_job()
