import json
import os
import random
import datetime

import pytest
import yaml

from invoke import run
from invoke.context import Context
from retrying import retry

import test.test_utils.eks as eks_utils
from src.github import GitHubHandler
from test.test_utils import is_pr_context, SKIP_PR_REASON


LOGGER = eks_utils.LOGGER


def test_eks_mxnet_multinode_training(mxnet_training, example_only):
    """Run MXNet distributed training on EKS using docker images with MNIST dataset"""
    random.seed(f"{mxnet_training}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 6000)
    namespace = f"mxnet-multi-node-training-{unique_id}"
    app_name = f"eks-mxnet-mnist-app-{unique_id}"
    job_name = f"kubeflow-mxnet-gpu-dist-job-{unique_id}"

    # TODO: This should either be dynamic or at least global variables
    num_workers = 3
    num_servers = 2
    gpu_limit = 1
    command = ["python"]
    args = [
        "/incubator-mxnet/example/image-classification/train_mnist.py",
        "--num-epochs",
        "20",
        "--num-layers",
        "2",
        "--kv-store",
        "dist_device_sync",
        "--gpus",
        "0",
    ]
    remote_yaml_file_path = "/tmp/mxnet_multi_node_training.yaml"

    generate_mxnet_multinode_yaml_file(
        mxnet_training, job_name, num_workers, num_servers, gpu_limit, command, args, remote_yaml_file_path
    )

    training_result = run_eks_mxnet_multi_node_training(namespace, app_name, job_name, remote_yaml_file_path, unique_id)
    assert training_result, "EKS multinode training failed"


def generate_mxnet_multinode_yaml_file(
    container_image, job_name, num_workers, num_servers, gpu_limit, command, args, remote_yaml_file_path
):
    """Function that writes the yaml file for a given container_image and commands to create a pod.
    Args:
        container_img, job_name, num_workers, num_servers, gpu_limit, command, args: list, remote_yaml_file_path: str
    """

    yaml_data = {
        "apiVersion": "kubeflow.org/v1alpha1",
        "kind": "MXJob",
        "metadata": {"name": job_name},
        "spec": {
            "jobMode": "dist",
            "replicaSpecs": [
                {
                    "replicas": 1,
                    "mxReplicaType": "SCHEDULER",
                    "PsRootPort": 9000,
                    "template": {
                        "spec": {
                            "containers": [{"name": "mxnet", "image": container_image}],
                            "restartPolicy": "OnFailure",
                        }
                    },
                },
                {
                    "replicas": num_servers,
                    "mxReplicaType": "SERVER",
                    "template": {"spec": {"containers": [{"name": "mxnet", "image": container_image}],}},
                },
                {
                    "replicas": num_workers,
                    "mxReplicaType": "WORKER",
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "mxnet",
                                    "image": container_image,
                                    "command": command,
                                    "args": args,
                                    "resources": {"limits": {"nvidia.com/gpu": gpu_limit}},
                                }
                            ],
                            "restartPolicy": "OnFailure",
                        }
                    },
                },
            ],
        },
    }
    with open(remote_yaml_file_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    LOGGER.info("Uploaded generated yaml file to %s", remote_yaml_file_path)


def run_eks_mxnet_multi_node_training(namespace, app_name, job_name, remote_yaml_file_path, unique_id):
    """Run MXNet distributed training on EKS using MXNet Operator
    Args:
    namespace, app_name, job_name, remote_yaml_file_path
    """

    kubeflow_version = "v0.4.1"
    home_dir = run("echo $HOME").stdout.strip("\n")
    path_to_ksonnet_app = os.path.join(home_dir, f"mxnet_multi_node_eks_test-{unique_id}")
    env = f"{namespace}-env"

    training_result = False

    ctx = Context()

    # Namespaces will allow parallel runs on the same cluster. Create namespace if it doesnt exist.
    does_namespace_exist = ctx.run("kubectl get namespace | grep {}".format(namespace), warn=True)
    if not does_namespace_exist:
        ctx.run("kubectl create namespace {}".format(namespace))
    if not os.path.exists(path_to_ksonnet_app):
        ctx.run(f"mkdir -p {path_to_ksonnet_app}")

    ctx.run("ks init {}".format(app_name))

    with ctx.cd(f"{path_to_ksonnet_app}"):
        ctx.run(f"rm -rf {app_name}")
        github_handler = GitHubHandler("aws", "kubeflow")
        github_token = github_handler.get_auth_token()
        ctx.run(f"ks init {app_name} --namespace {namespace}")

        with ctx.cd(app_name):
            ctx.run(f"ks env add {env} --namespace {namespace}")
            # Check if the kubeflow registry exists and create. Registry will be available in each pod.
            does_registry_exist = ctx.run("ks registry list | grep kubeflow", warn=True)
            if not does_registry_exist:
                ctx.run(
                    "ks registry add kubeflow github.com/kubeflow/kubeflow/tree/{}/kubeflow".format(kubeflow_version),
                    env={"GITHUB_TOKEN": github_token},
                    hide=True,
                )
                ctx.run(
                    "ks pkg install kubeflow/mxnet-job@{}".format(kubeflow_version),
                    env={"GITHUB_TOKEN": github_token},
                    hide=True,
                )

                ctx.run("ks generate mxnet-operator mxnet-operator", hide=True)

                try:
                    ctx.run(f"kubectl get pods -n {namespace} -o wide")
                    LOGGER.debug(f"ks apply {env} -c mxnet-operator -n {namespace}")
                    ctx.run(f"ks apply {env} -c mxnet-operator -n {namespace}")
                    # Delete old job with same name if exists
                    ctx.run(f"kubectl delete -f {remote_yaml_file_path}", warn=True)
                    ctx.run(f"kubectl create -f {remote_yaml_file_path} -n {namespace}")
                    if is_mxnet_eks_multinode_training_complete(job_name, namespace):
                        training_result = True
                finally:
                    eks_utils.eks_multinode_cleanup("", job_name, namespace, env)

    return training_result


@retry(stop_max_attempt_number=40, wait_fixed=6000, retry_on_exception=eks_utils.retry_if_value_error)
def is_mxnet_eks_multinode_training_complete(job_name, namespace):
    """Function to check job and pod status for multinode training.
    A separate method is required because kubectl commands for logs and status are different with namespaces.
    Args:
        job_name: str, remote_yaml_file_path: str
    """
    run_out = run(f"kubectl get mxjobs -n {namespace} {job_name} -o json", warn=True)
    pod_info = json.loads(run_out.stdout)

    if "status" not in pod_info:
        raise ValueError("Waiting for job to launch...")

    # Job_phase can be one of the Creating, Running, Cleanup, Failed, Done
    # Job state can be one of the Running, Succeeded, Failed
    if "phase" in pod_info["status"]:
        job_phase = pod_info["status"]["phase"]
        job_state = pod_info["status"]["state"]
        LOGGER.info("Current job phase: %s", job_phase)

        if "Failed" in job_state:
            LOGGER.info("Failure: Job failed to run and the pods are getting terminated.")
        elif "Succeeded" in job_state:
            if "Done" in job_phase or "CleanUp" in job_phase:
                LOGGER.info("SUCCESS: Job is complete. Pods are getting terminated.")
                return True
        elif "Running" in job_state:
            if "Creating" in job_phase:
                LOGGER.info("IN-PROGRESS: Container is either Creating. Waiting to complete...")
                raise ValueError("IN-PROGRESS: Container getting created.")
            elif "Running" in job_phase:
                # Print logs generated
                run(
                    "kubetail $(kubectl get pods | grep {} | cut -f 1 -d ' ' | paste -s -d, -) --follow "
                    "false".format(job_name + "-worker"),
                    warn=True,
                )
                raise ValueError("IN-PROGRESS: Job is running.")
            elif "CleanUp" in job_phase or "Failed" in job_phase:
                LOGGER.info("Failed: The job failed to execute. Pods are getting terminated.")
            elif "Done" in job_phase:
                LOGGER.info("Failed: The job failed to execute. Pods are getting terminated.")

    return False
