"""
Helper functions for EKS Integration Tests
"""

import os
import sys
import json
import logging
import random
import re

import boto3

from botocore.exceptions import ClientError
from retrying import retry
from invoke import run, Context

DEFAULT_REGION = "us-west-2"

# Path till directory test/
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Use as prefix for file paths in ec2, ecs and eks tests
DLC_TESTS_PREFIX = os.path.join(os.sep, ROOT_DIR, "dlc_tests")

SINGLE_NODE_TRAINING_TEMPLATE_PATH = os.path.join(
    os.sep,
    DLC_TESTS_PREFIX,
    "eks",
    "eks_manifest_templates",
    "training",
    "single_node_training.yaml",
)

SINGLE_NODE_INFERENCE_TEMPLATE_PATH = os.path.join(
    os.sep,
    DLC_TESTS_PREFIX,
    "eks",
    "eks_manisfest_templates"
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def get_aws_secret_yml_path():

    return os.path.join(
        os.sep,
        DLC_TESTS_PREFIX,
        "eks",
        "eks_manifest_templates",
        "aws_access",
        "secret.yaml",
    )

def get_single_node_training_template_path():

    return os.path.join(
        os.sep,
        DLC_TESTS_PREFIX,
        "eks",
        "eks_manifest_templates",
        "training",
        "single_node_training.yaml",
    )

def get_single_node_inference_template_path(framework, processor):

    return os.path.join(
        os.sep,
        DLC_TESTS_PREFIX,
        "eks",
        "eks_manifest_templates",
        framework,
        "inference",
        f"single_node_{processor}_inference.yaml",
    )


def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)


@retry(
    stop_max_attempt_number=360,
    wait_fixed=10000,
    retry_on_exception=retry_if_value_error,
)
def is_eks_training_complete(pod_name):
    """Function to check if the pod status has reached 'Completion'
    Args:
        pod_name: str
    """

    run_out = run("kubectl get pod {} -o json".format(pod_name))
    pod_info = json.loads(run_out.stdout)

    if "containerStatuses" in pod_info["status"]:
        container_status = pod_info["status"]["containerStatuses"][0]
        LOGGER.info("Container Status: %s", container_status)
        if container_status["name"] == pod_name:
            if "terminated" in container_status["state"]:
                if container_status["state"]["terminated"]["reason"] == "Completed":
                    LOGGER.info("SUCCESS: The container terminated.")
                    return True
                elif container_status["state"]["terminated"]["reason"] == "Error":
                    error_out = run("kubectl logs {}".format(pod_name)).stdout
                    # delete pod in case of error
                    run("kubectl delete pods {}".format(pod_name))
                    LOGGER.error(
                        "ERROR: The container run threw an error and terminated. "
                        "kubectl logs: %s",
                        error_out,
                    )
                    raise AttributeError("Container Error!")
            elif (
                "waiting" in container_status["state"]
                and container_status["state"]["waiting"]["reason"] == "CrashLoopBackOff"
            ):
                error_out = run("kubectl logs {}".format(pod_name)).stdout
                # delete pod in case of error
                run("kubectl delete pods {}".format(pod_name))
                LOGGER.error(
                    "ERROR: The container run threw an error in waiting state. "
                    "kubectl logs: %s",
                    error_out,
                )
                raise AttributeError("Error: CrashLoopBackOff!")
            elif (
                "waiting" in container_status["state"]
                or "running" in container_status["state"]
            ):
                LOGGER.info(
                    "IN-PROGRESS: Container is either Creating or Running. Waiting to complete..."
                )
                raise ValueError("IN-PROGRESS: Retry.")
    else:
        LOGGER.info(f"containerStatuses not available yet, retrying. Pod: {pod_name}")
        raise ValueError("IN-PROGRESS: Retry.")

    return False

def write_eks_yaml_file_from_template(
    local_template_file_path, remote_yaml_file_path, search_replace_dict
):
    """Function that does a simple replace operation based on the search_replace_dict on the template file contents
    and writes the final yaml file to remote_yaml_path
    Args:
        local_template_path, remote_yaml_path: str
        search_replace_dict: dict
    """
    with open(local_template_file_path, "r") as yaml_file:
        yaml_data = yaml_file.read()

    for key, value in search_replace_dict.items():
        yaml_data = yaml_data.replace(key, value)

    with open(remote_yaml_file_path, "w") as yaml_file:
        yaml_file.write(yaml_data)

    LOGGER.info("Copied generated yaml file to %s", remote_yaml_file_path)


def is_eks_cluster_active(eks_cluster_name):
    """Function to verify if the default eks cluster is up and running.
    Args:
        eks_cluster_name: str
    Return:
        if_active: bool, true if status is active
    """
    if_active = False

    eksctl_check_cluster_command = """eksctl get cluster {} -o json
    """.format(
        eks_cluster_name
    )

    run_out = run(eksctl_check_cluster_command, warn=True)

    if run_out.return_code == 0:
        cluster_info = json.loads(run_out.stdout)[0]
        if_active = cluster_info["Status"] == "ACTIVE"

    return if_active


def get_eks_role():
    return os.getenv("EKS_TEST_ROLE")

def eks_write_kubeconfig(eks_cluster_name, region="us-west-2"):
    """Function that writes the aws eks configuration for the specified cluster in the file ~/.kube/config
    This file is used by the kubectl and ks utilities along with aws-iam-authenticator to authenticate with aws
    and query the eks cluster.
    Note: This function assumes the cluster is 'ACTIVE'. Please use check_eks_cluster_status() to obtain status
    of the cluster.
    Args:
        eks_cluster_name, region: str
    """
    eks_role = get_eks_role()
    eksctl_write_kubeconfig_command = f"eksctl utils write-kubeconfig --name {eks_cluster_name} --region {region}"
    
    if eks_role:
        eksctl_write_kubeconfig_command += f" --authenticator-role-arn {eks_role} "
    
    run(eksctl_write_kubeconfig_command)

    run("cat /root/.kube/config", warn=True)


def eks_forward_port_between_host_and_container(selector_name, host_port, container_port, namespace="default"):
    """Uses kubectl port-forward command to forward a port from the container pods to the host.
    Note: The 'host' in this case is the gateway host, and not the worker hosts.
    Args:
        namespace, selector_name: str
        host_port, container_port: int
    """

    # Terminate other port-forwards
    # run("lsof -ni | awk '{print $2}' |  grep -v PID | uniq | xargs kill -9", warn=True)

    run("nohup kubectl port-forward -n {0} `kubectl get pods -n {0} --selector=app={1} -o "
        "jsonpath='{{.items[0].metadata.name}}'` {2}:{3} > /dev/null 2>&1 &".format(namespace, selector_name, host_port, container_port))


@retry(stop_max_attempt_number=20, wait_fixed=30000, retry_on_exception=retry_if_value_error)
def is_service_running(selector_name, namespace="default"):
    """Check if the service pod is running
    Args:
        namespace, selector_name: str
    """
    run_out = run("kubectl get pods -n {} --selector=app={} -o jsonpath='{{.items[0].status.phase}}' ".format(namespace, selector_name), warn=True)

    if run_out.stdout == "Running":
        return True
    else:
        raise ValueError("Service not running yet, try again")

def eks_multinode_cleanup(remote_yaml_file_path, namespace):
    """
    Function to cleanup jobs created by EKS
    :param namespace:
    :param remote_yaml_file_path:
    :param namespace:
    :return:
    """

    run(f"kubectl delete -f {remote_yaml_file_path} -n {namespace}", warn=True)


def eks_multinode_get_logs(namespace, pod_name):
    """
    Function to get logs for a pod in the specified namespace.
    :param namespace:
    :param pod_name:
    :return:
    """
    return run(f"kubectl logs -n {namespace} -f {pod_name}").stdout


@retry(stop_max_attempt_number=120, wait_fixed=10000, retry_on_exception=retry_if_value_error)
def is_mpijob_launcher_pod_ready(namespace, job_name):
    """Check if the MpiJob Launcher Pod is Ready
    Args:
        namespace: str
        job_name: str
    """

    pod_name = run(
        f"kubectl get pods -n {namespace} -l mpi_job_name={job_name},mpi_role_type=launcher -o name"
    ).stdout.strip("\n")
    if pod_name:
        return pod_name
    else:
        raise ValueError("Launcher pod is not ready yet, try again.")


@retry(stop_max_attempt_number=40, wait_fixed=60000, retry_on_exception=retry_if_value_error)
def is_eks_multinode_training_complete(remote_yaml_file_path, namespace, pod_name, job_name):
    """Function to check if the pod status has reached 'Completion' for multinode training.
    A separate method is required because kubectl commands for logs and status are different with namespaces.
    Args:
        remote_yaml_file_path, namespace, pod_name, job_name: str
    """

    run_out = run(f"kubectl get pod -n {namespace} {pod_name} -o json")
    pod_info = json.loads(run_out.stdout)

    if 'containerStatuses' in pod_info['status']:
        container_status = pod_info['status']['containerStatuses'][0]
        LOGGER.info(f"Container Status: {container_status}")
        if container_status['name'] == job_name:
            if "terminated" in container_status['state']:
                if container_status['state']['terminated']['reason'] == "Completed":
                    LOGGER.info("SUCCESS: The container terminated.")
                    return True
                elif container_status['state']['terminated']['reason'] == "Error":
                    LOGGER.error(f"ERROR: The container run threw an error and terminated. "
                                 f"kubectl logs: {eks_multinode_get_logs(namespace, pod_name)}")
                    eks_multinode_cleanup(remote_yaml_file_path, namespace)
                    raise AttributeError("Container Error!")
            elif 'waiting' in container_status['state'] and \
                    container_status['state']['waiting']['reason'] == "PodInitializing":
                LOGGER.info("POD-INITIALIZING: Pod is initializing")
                raise ValueError("IN-PROGRESS: Retry.")
            elif 'waiting' in container_status['state'] and \
                    container_status['state']['waiting']['reason'] == "CrashLoopBackOff":
                LOGGER.error(f"ERROR: The container run threw an error in waiting state. "
                             f"kubectl logs: {eks_multinode_get_logs(namespace, pod_name)}")
                eks_multinode_cleanup(remote_yaml_file_path, namespace)
                raise AttributeError("Error: CrashLoopBackOff!")
            elif 'waiting' in container_status['state'] or 'running' in container_status['state']:
                LOGGER.info("IN-PROGRESS: Container is either Creating or Running. Waiting to complete...")
                raise ValueError("IN-PROGRESS: Retry.")

    return False


def get_dgl_branch(ctx, image_uri):
    """
    Determine which dgl git branch to use based on the latest version

    :param ctx: Invoke context
    :param image_uri: docker image URI, used to uniqify repo name to avoid asynchronous git pulls
    :return: latest dgl branch, i.e. 0.5.x
    """
    image_addition = image_uri.split('/')[-1].replace(':', '-')
    dgl_local_repo = f'.dgl_branch-{image_addition}'
    ctx.run(f"git clone https://github.com/dmlc/dgl.git {dgl_local_repo}", hide=True, warn=True)
    with ctx.cd(dgl_local_repo):
        branch = ctx.run("git branch -r", hide=True)
        branches = branch.stdout.split()
        release_branch_regex = re.compile(r'\d+.\d+.x')
        release_branches = []
        for branch in branches:
            match = release_branch_regex.search(branch)
            if match:
                release_branches.append(match.group())
    release_branches = sorted(release_branches, reverse=True)
    return release_branches[0]
