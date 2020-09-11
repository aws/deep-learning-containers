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


EKS_VERSION = "1.17.9"
KUBETAIL_VERSION = "1.6.7"

SSH_PUBLIC_KEY_NAME = "dlc-ec2-keypair-prod"
PR_EKS_CLUSTER_NAME_TEMPLATE = "dlc-eks-pr-{}-test-cluster"

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


def init_cfn_client():
    """Function to initiate the cfn session
    Args:
        material_set: str
    """
    return boto3.client('cloudformation')

def init_iam_client():
    """Function to initiate the iam session
    Args:
        material_set: str
    """
    return boto3.client('iam')

def init_eks_client():
    """Function to initiate the eks session
    Args:
        material_set: str
    """
    return boto3.client('eks', region_name=DEFAULT_REGION)

def list_cfn_stack_names():
    """Function to list the cfn stacks in the account.
    Note: lists all the cfn stacks that aren't
    Args:
        material_set: str
    """
    stack_statuses = ['CREATE_IN_PROGRESS', 'CREATE_FAILED', 'CREATE_COMPLETE', 'ROLLBACK_IN_PROGRESS',
                      'ROLLBACK_FAILED', 'ROLLBACK_COMPLETE', 'DELETE_IN_PROGRESS', 'DELETE_FAILED',
                      'UPDATE_IN_PROGRESS', 'UPDATE_COMPLETE_CLEANUP_IN_PROGRESS', 'UPDATE_COMPLETE',
                      'UPDATE_ROLLBACK_IN_PROGRESS', 'UPDATE_ROLLBACK_FAILED',
                      'UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS', 'UPDATE_ROLLBACK_COMPLETE',
                      'REVIEW_IN_PROGRESS', 'DELETE_COMPLETE']
    cfn = init_cfn_client()

    try:
        cfn_stacks = cfn.list_stacks(
            StackStatusFilter=[status for status in stack_statuses if status != 'DELETE_COMPLETE']
        )
    except ClientError as e:
        LOGGER.error(f"Error: Cannot list stack names. Full Exception:\n{e}")

    return [stack['StackName'] for stack in cfn_stacks['StackSummaries']]


def describe_cfn_stack_events(stack_name):
    """
    Function to describe CFN events.
    Args:
        stack_name, materialset: str
    """
    cfn = init_cfn_client()
    max_items = 10
    try:
        LOGGER.info("Describing the latest {} events on the stack".format(max_items))
        for stack_event in cfn.describe_stack_events(StackName=stack_name)["StackEvents"][:max_items]:
            LOGGER.info(stack_event)
    except ClientError as e:
        LOGGER.error(f"Error: Cannot describe events on stack: {stack_name}. Full Exception:\n{e}")

def eks_setup():
    """Function to download kubectl, aws-iam-authenticator and ksonnet binaries
    Utilities:
    1. kubectl: create and manage runs on eks cluster
    2. aws-iam-authenticator: authenticate the instance to access eks with the appropriate aws credentials
    """

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run_out = run(
        "kubectl version --short --client && aws-iam-authenticator version",
        warn=True,
    )

    eks_tools_installed = not run_out.return_code

    if eks_tools_installed:
        return

    platform = run("uname -s").stdout.strip()

    kubectl_download_command = (
        f"curl --silent --location https://amazon-eks.s3-us-west-2.amazonaws.com/"
        f"{EKS_VERSION}/2020-08-04/bin/{platform.lower()}/amd64/kubectl -o /usr/local/bin/kubectl"
    )

    aws_iam_authenticator_download_command = (
        f"curl --silent --location https://amazon-eks.s3-us-west-2.amazonaws.com/"
        f"{EKS_VERSION}/2020-08-04/bin/{platform.lower()}/amd64/aws-iam-authenticator "
        f"-o /usr/local/bin/aws-iam-authenticator"
    )

    kubetail_download_command = (
        f"curl --silent --location https://raw.githubusercontent.com/johanhaleby/kubetail/"
        f"{KUBETAIL_VERSION}/kubetail -o /usr/local/bin/kubetail"
    )

    run(kubectl_download_command, echo=True)
    run("chmod +x /usr/local/bin/kubectl")

    run(aws_iam_authenticator_download_command, echo=True)
    run("chmod +x /usr/local/bin/aws-iam-authenticator")

    run(kubetail_download_command, echo=True)
    run("chmod +x /usr/local/bin/kubetail")

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run("kubectl version --short --client", echo=True)
    run("aws-iam-authenticator version", echo=True)

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

def get_iam_role_arn(role_name):
    iam_client = init_iam_client()
    return iam_client.get_role(RoleName=role_name)['Role']['Arn']


def eks_write_kubeconfig(eks_cluster_name, region="us-west-2"):
    """Function that writes the aws eks configuration for the specified cluster in the file ~/.kube/config
    This file is used by the kubectl and kfctl utilities along with aws-iam-authenticator to authenticate with aws
    and query the eks cluster.
    Args:
        eks_cluster_name, region: str
    """

    iam_client = init_iam_client()
    eks_role = get_iam_role_arn('eksClusterAccess')
    write_kubeconfig_command = f"aws eks update-kubeconfig --name {eks_cluster_name} --region {region} --role-arn {eks_role}"
    run(write_kubeconfig_command)

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

def delete_eks_nodegroup(eks_cluster_name, nodegroup_name):
    """ Function to delete EKS NodeGroup
        :param eks_cluster_name: Name of the EKS cluster
        :param nodegroup_name: Name of the nodegroup attached to the cluster
        :param region: Region where EKS cluster is located
        :return: None
    """
    eks_client = init_eks_client()

    eks_client.delete_nodegroup(
    clusterName=eks_cluster_name,
    nodegroupName=nodegroup_name
)

    nodegroup_waiter(eks_cluster_name, nodegroup_name, 'nodegroup_deleted')

    LOGGER.info("EKS cluster nodegroup deleted successfully, with the following parameters\n"
                f"cluster_name: {eks_cluster_name}\n"
                f"nodegroup_name: {nodegroup_name}")


def describe_eks_cluster(eks_cluster_name):
    """ Function to describe EKS cluster
        :param eks_cluster_name: Name of the EKS cluster
        :return: information about EKS cluster 
    """

    eks_client = init_eks_client()

    try:
        response = eks_client.describe_cluster(name=eks_cluster_name)
    except Exception as e:
        LOGGER.error(f"Error: Cannot describe EKS cluster {eks_cluster_name}. Full Exception:\n{e}")
    return response['cluster']

def check_eks_cluster_state(eks_cluster_name):
    """ Function to check the state EKS cluster
        :param eks_cluster_name: Name of the EKS cluster
        :return: boolean  
    """
    response = describe_eks_cluster(eks_cluster_name)
    return response['status'] == 'ACTIVE'


def get_eks_cluster_security_group(eks_cluster_name):
    """ Function to get security group attached to the EKS cluster 
        :param eks_cluster_name: Name of the EKS cluster
        :return: security group id
    """
    response = describe_eks_cluster(eks_cluster_name)
    return response['resourcesVpcConfig']['clusterSecurityGroupId']

def get_eks_cluster_subnet(eks_cluster_name):
    """ Function to get list of subnets configured for EKS cluster 
        :param eks_cluster_name: Name of the EKS cluster
        :return: vpc subnets
    """
    response = describe_eks_cluster(eks_cluster_name)
    return response['resourcesVpcConfig']['subnetIds']

def create_eks_cluster_nodegroup(
        eks_cluster_name, nodegroup_name, num_nodes, instance_type, ssh_public_key_name, region=DEFAULT_REGION
):
    """
    Function to create and attach a nodegroup to an existing EKS cluster.
    :param eks_cluster_name: Cluster name of the form PR_EKS_CLUSTER_NAME_TEMPLATE
    :nodegroup_name: nodegroup name
    :param num_nodes: number of nodes to create in nodegroup
    :param instance_type: instance type to use for nodegroup instances
    :param ssh_public_key_name:
    :param region: Region where EKS cluster is located
    :return: None
    """
    eks_client = init_eks_client()

    nodegroup_role = get_iam_role_arn('eks_nodegroup_role')
    security_group = get_eks_cluster_security_group(eks_cluster_name)
    subnet_id = get_eks_cluster_subnet(eks_cluster_name)

    eks_client.create_nodegroup(
    clusterName=eks_cluster_name,
    nodegroupName=nodegroup_name,
    scalingConfig={
        'minSize': num_nodes,
        'maxSize': num_nodes,
        'desiredSize': num_nodes
    },
    subnets=subnet_id,
    instanceTypes=[
        instance_type
    ],
    amiType='AL2_x86_64_GPU',
    remoteAccess={
        'ec2SshKey': ssh_public_key_name,
        'sourceSecurityGroups': [
            security_group
        ]
    },
    nodeRole=nodegroup_role,
)
    nodegroup_waiter(eks_cluster_name, nodegroup_name, 'nodegroup_active')

    LOGGER.info("EKS cluster nodegroup created successfully, with the following parameters\n"
                f"cluster_name: {eks_cluster_name}\n"
                f"num_nodes: {num_nodes}\n"
                f"instance_type: {instance_type}\n"
                f"ssh_public_key: {ssh_public_key_name}")


def nodegroup_waiter(eks_cluster_name, nodegroup_name, action):
    """ Wait for the nodegroup to be active
        :param eks_cluster_name: Name of the EKS cluster
        :nodegroup_name: nodegroup name
        :param action: boto3 waiter action
        :return: None  
    """
    eks_client = init_eks_client()
    _delay = 60
    _max_attempts = 20
    try:
        waiter = eks_client.get_waiter(action)
        waiter.wait(
            clusterName=eks_cluster_name,
            nodegroupName=nodegroup_name,
            WaiterConfig={
                'Delay': _delay,
                'MaxAttempts': _max_attempts
            })
    except ClientError as e:
        LOGGER.error(f"Error: Cannot create/delete nodegroup: {nodegroup_name}. Full Exception:\n{e}")

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
    :return: latest dgl branch, i.e. 0.4.x
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
