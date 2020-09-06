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


EKS_VERSION = "1.14.6"
EKSCTL_VERSION = "0.22.0"
KFCTL_VERSION = "v1.0.2"
KUBEFLOW_VERSION = "v0.4.1"
KUBETAIL_VERSION = "1.6.7"

EKS_NVIDIA_PLUGIN_VERSION = "0.6.0"

# https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html
EKS_AMI_ID = {"cpu": "ami-03086423d09685de3", "gpu": "ami-061798711b2adafb4"}

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


def delete_cfn_stack_and_wait(stack_name):
    """Function to delete cfn stack. The waiter checks if the stack has been deleted every _delay seconds,
    for a maximum of _max_attempts times i.e. for _max_attempts min.
    Args:
        stack_name, material_set: str
    """
    cfn = init_cfn_client()
    _delay = 60
    _max_attempts = 20
    try:
        cfn.delete_stack(StackName=stack_name)
        cfn_waiter = cfn.get_waiter("stack_delete_complete")
        cfn_waiter.wait(StackName=stack_name,
                        WaiterConfig={
                            'Delay': _delay,
                            'MaxAttempts': _max_attempts
                        })
    except ClientError as e:
        LOGGER.error(f"Error: Cannot delete stack: {stack_name}. Full Exception:\n{e}")
        describe_cfn_stack_events(stack_name)

def delete_oidc_provider(eks_cluster_name):
    """Function to delete the oidc provider created by kubeflow
    Args:
        eks_cluster_name: str
    """
    iam_client = boto3.client('iam')
    eks_client = boto3.client('eks', region_name=DEFAULT_REGION)
    sts_client = boto3.client('sts')

    try:
        account_id = sts_client.get_caller_identity().get('Account')
        response = eks_client.describe_cluster(name=eks_cluster_name)        
        oidc_issuer = response['cluster']['identity']['oidc']['issuer']
        oidc_url = oidc_issuer.rsplit('//', 1)[-1]
        oidc_provider_arn = f"arn:aws:iam::{account_id}:oidc-provider/{oidc_url}"

        LOGGER.info(f"Deleting oidc provider: {oidc_provider_arn}")
        iam_client.delete_open_id_connect_provider(OpenIDConnectProviderArn=oidc_provider_arn)
        LOGGER.info(f"Deleting IAM roles created by kubeflow")
        delete_iam_roles(eks_cluster_name)

    except ClientError as e:
        LOGGER.error(f"Error: Cannot describe the EKS cluster: {eks_cluster_name}. Full Exception:\n{e}")

def delete_iam_roles(eks_cluster_name):
    """Function to delete IAM role and policy created by kubeflow
    """
    iam_resource = boto3.resource('iam')
    role_list = [f'kf-admin-{eks_cluster_name}', f'kf-user-{eks_cluster_name}']
    try:
        for role in role_list:
            iam_role = iam_resource.Role(name=role)
            
            for role_policy in iam_role.policies.all():
                LOGGER.info(f"Deleting Policy {role_policy.name}")
                role_policy.delete()

            iam_role.delete()
            LOGGER.info(f"IAM role {iam_role.name} deleted\n")
    except ClientError as e:
        LOGGER.error(f"Error: Cannot delete IAM role. Full Exception:\n{e}")

def delete_eks_cluster(eks_cluster_name):
    """Function to delete the EKS cluster, if it exists. Additionally, the function cleans up the oidc provider
    created by kubeflow and any cloudformation stacks that are dangling. 
    Args:
        eks_cluster_name: str
    """
    
    delete_oidc_provider(eks_cluster_name)

    run("eksctl delete cluster {} --wait".format(eks_cluster_name), warn=True)

    cfn_stack_names = list_cfn_stack_names()
    for stack_name in cfn_stack_names:
        if eks_cluster_name in stack_name:
            LOGGER.info(f"Deleting dangling cloudformation stack: {stack_name}")
            delete_cfn_stack_and_wait(stack_name)

def setup_eksctl():
    run_out = run("eksctl version", echo=True, warn=True)

    eksctl_installed = not run_out.return_code

    if eksctl_installed:
        return

    platform = run("uname -s", echo=True).stdout.strip()
    eksctl_download_command = (
        f"curl --silent --location https://github.com/weaveworks/eksctl/releases/download/"
        f"{EKSCTL_VERSION}/eksctl_{platform}_amd64.tar.gz | tar xz -C /tmp"
    )
    run(eksctl_download_command, echo=True)
    run("mv /tmp/eksctl /usr/local/bin")


@retry(stop_max_attempt_number=2, wait_fixed=60000)
def create_eks_cluster(eks_cluster_name, processor_type, num_nodes,
                       instance_type, ssh_public_key_name, region=os.getenv("AWS_REGION", DEFAULT_REGION)):
    """Function to setup an EKS cluster using eksctl. The AWS credentials used to perform eks operations
    are that the user deepamiuser-beta as used in other functions. The 'deeplearning-ami-beta' public key
    will be used to access the nodes created as EC2 instances in the EKS cluster.
    Note: eksctl creates a cloudformation stack by the name of eksctl-${eks_cluster_name}-cluster.
    Args:
        eks_cluster_name, processor_type, num_nodes, instance_type, ssh_public_key_name: str
    """
    setup_eksctl()

    delete_eks_cluster(eks_cluster_name)

    eksctl_create_cluster_command = f"eksctl create cluster {eks_cluster_name} " \
                                    f"--node-ami {EKS_AMI_ID[processor_type]} " \
                                    f"--nodes {num_nodes} " \
                                    f"--node-type={instance_type} " \
                                    f"--timeout=40m " \
                                    f"--ssh-access " \
                                    f"--ssh-public-key {ssh_public_key_name} " \
                                    f"--region {region}"

    # In us-east-1 you are likely to get UnsupportedAvailabilityZoneException,
    # if the allocated zones is us-east-1e as it does not support AmazonEKS
    if region == "us-east-1":
        eksctl_create_cluster_command += " --zones=us-east-1a,us-east-1b,us-east-1d "
    eksctl_create_cluster_command += " --auto-kubeconfig "
    run(eksctl_create_cluster_command)

    eks_write_kubeconfig(eks_cluster_name, "us-west-2")
    
    LOGGER.info(f"EKS cluster created successfully, with the following parameters cluster_name: "
                f"{eks_cluster_name} ami-id: {EKS_AMI_ID[processor_type]} num_nodes: {num_nodes} instance_type: "
                f"{instance_type} ssh_public_key: {ssh_public_key_name}")


def eks_setup():
    """Function to download eksctl, kubectl, aws-iam-authenticator and ksonnet binaries
    Utilities:
    1. eksctl: create and manage cluster
    2. kubectl: create and manage runs on eks cluster
    3. aws-iam-authenticator: authenticate the instance to access eks with the appropriate aws credentials
    4. kfctl: control plane for deploying and managing Kubeflow
    """

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run_out = run(
        "eksctl version && kubectl version --short --client && aws-iam-authenticator version && kfctl version",
        warn=True,
    )

    eks_tools_installed = not run_out.return_code

    if eks_tools_installed:
        return

    platform = run("uname -s").stdout.strip()

    kubectl_download_command = (
        f"curl --silent --location https://amazon-eks.s3-us-west-2.amazonaws.com/"
        f"{EKS_VERSION}/2019-08-22/bin/{platform.lower()}/amd64/kubectl -o /usr/local/bin/kubectl"
    )

    aws_iam_authenticator_download_command = (
        f"curl --silent --location https://amazon-eks.s3-us-west-2.amazonaws.com/"
        f"{EKS_VERSION}/2019-08-22/bin/{platform.lower()}/amd64/aws-iam-authenticator "
        f"-o /usr/local/bin/aws-iam-authenticator"
    )

    kfctl_download_command = (
        f"curl --silent --location https://github.com/kubeflow/kfctl/releases/download/{KFCTL_VERSION}/kfctl_{KFCTL_VERSION}-0-ga476281_{platform.lower()}.tar.gz "
        f"-o /tmp/kfctl_{KFCTL_VERSION}_{platform.lower()}.tar.gz"
    )

    kubetail_download_command = (
        f"curl --silent --location https://raw.githubusercontent.com/johanhaleby/kubetail/"
        f"{KUBETAIL_VERSION}/kubetail -o /usr/local/bin/kubetail"
    )

    # Separate function handles setting up eksctl
    setup_eksctl()

    run(kubectl_download_command, echo=True)
    run("chmod +x /usr/local/bin/kubectl")

    run(aws_iam_authenticator_download_command, echo=True)
    run("chmod +x /usr/local/bin/aws-iam-authenticator")

    run(kfctl_download_command, echo=True)
    run(f"tar -xvf /tmp/kfctl_{KFCTL_VERSION}_{platform.lower()}.tar.gz -C /tmp --strip-components=1")
    run("mv /tmp/kfctl /usr/local/bin")

    run(kubetail_download_command, echo=True)
    run("chmod +x /usr/local/bin/kubetail")

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run("eksctl version", echo=True)
    run("kubectl version --short --client", echo=True)
    run("aws-iam-authenticator version", echo=True)
    run("kfctl version", echo=True)


def setup_kubeflow(eks_cluster_name,region=os.getenv("AWS_REGION", DEFAULT_REGION)):
    """Function to setup kubeflow v1.0.2
        The mxnet operator configuration is not included in the kubeflow v1.0.2 hence installing manually.
        The mpi operator included in kubeflow v1.0.2 has version v0.1 which has known issues in EKS hence installing the latest
        version available v0.2
    """

    unique_id = random.randint(1, 6000)
    local_template_file_path = os.path.join(
        "eks",
        "eks_manifest_templates",
        "kubeflow",
        "kfctl_aws_v1.0.2.yaml"
    )
    run(f"mkdir -p /tmp/{eks_cluster_name}")

    remote_yaml_path = os.path.join(os.sep, "tmp", eks_cluster_name, f"kfctl_aws_{unique_id}.yaml")
    replace_dict = {
        "<REGION>": region
    }
    
    write_eks_yaml_file_from_template(local_template_file_path, remote_yaml_path, replace_dict)

    run(f"kfctl apply -V -f {remote_yaml_path}",echo=True)

    deploy_mxnet_operator()
    deploy_mpi_operator()


def deploy_mxnet_operator():
    """Function to deploy mxnet operator in the EKS cluster. This will support v1beta1 crd for mxjobs.
    """
    ctx = Context()
    home_dir = ctx.run("echo $HOME").stdout.strip("\n")
    mxnet_operator_dir = os.path.join(home_dir, "mxnet-operator")
    if os.path.isdir(mxnet_operator_dir):
        ctx.run(f"rm -rf {mxnet_operator_dir}")

    clone_mxnet_command = f"git clone https://github.com/kubeflow/mxnet-operator.git {mxnet_operator_dir}"
    ctx.run(clone_mxnet_command, echo=True)
    run(f"kubectl create -k {mxnet_operator_dir}/manifests/overlays/v1beta1/", echo=True)


def deploy_mpi_operator():
    """Function to deploy mpi operator in the EKS cluster. This will support v1alpha2 crd for mpijobs.
    """
    ctx = Context()
    home_dir = ctx.run("echo $HOME").stdout.strip("\n")
    mpi_operator_dir = os.path.join(home_dir, "mpi-operator")
    if os.path.isdir(mpi_operator_dir):
        ctx.run(f"rm -rf {mpi_operator_dir}")

    clone_mxnet_command = f"git clone https://github.com/kubeflow/mpi-operator {mpi_operator_dir}"
    run(clone_mxnet_command, echo=True)
    run(f"kubectl create -f {mpi_operator_dir}/deploy/v1alpha2/mpi-operator.yaml", echo=True)


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


def eks_write_kubeconfig(eks_cluster_name, region="us-west-2"):
    """Function that writes the aws eks configuration for the specified cluster in the file ~/.kube/config
    This file is used by the kubectl and ks utilities along with aws-iam-authenticator to authenticate with aws
    and query the eks cluster.
    Note: This function assumes the cluster is 'ACTIVE'. Please use check_eks_cluster_status() to obtain status
    of the cluster.
    Args:
        eks_cluster_name, region: str
    """
    eksctl_write_kubeconfig_command = f"eksctl utils write-kubeconfig --name {eks_cluster_name} --region {region}"
    run(eksctl_write_kubeconfig_command)

    # run(f"aws eks --region us-west-2 update-kubeconfig --name {eks_cluster_name} --kubeconfig /root/.kube/config --role-arn arn:aws:iam::669063966089:role/nikhilsk-eks-test-role")

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


def create_eks_cluster_nodegroup(
        eks_cluster_name, processor_type, num_nodes, instance_type, ssh_public_key_name, region=DEFAULT_REGION
):
    """
    Function to create and attach a nodegroup to an existing EKS cluster.
    :param eks_cluster_name: Cluster name of the form PR_EKS_CLUSTER_NAME_TEMPLATE
    :param processor_type: cpu/gpu
    :param num_nodes: number of nodes to create in nodegroup
    :param instance_type: instance type to use for nodegroup instances
    :param ssh_public_key_name:
    :param region: Region where EKS cluster is located
    :return: None
    """
    eksctl_create_nodegroup_command = (
        f"eksctl create nodegroup "
        f"--cluster {eks_cluster_name} "
        f"--node-ami {EKS_AMI_ID.get(processor_type)} "
        f"--nodes {num_nodes} "
        f"--node-type={instance_type} "
        f"--timeout=40m "
        f"--ssh-access "
        f"--ssh-public-key {ssh_public_key_name} "
        f"--region {region}"
    )

    run(eksctl_create_nodegroup_command)

    LOGGER.info("EKS cluster nodegroup created successfully, with the following parameters\n"
                f"cluster_name: {eks_cluster_name}\n"
                f"ami-id: {EKS_AMI_ID[processor_type]}\n"
                f"num_nodes: {num_nodes}\n"
                f"instance_type: {instance_type}\n"
                f"ssh_public_key: {ssh_public_key_name}")


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
