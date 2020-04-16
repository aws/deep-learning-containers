"""
Helper functions for EKS Integration Tests
"""

import os
import sys
import json
import logging

from retrying import retry
from invoke import run, Context

from src.github import GitHubHandler


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


EKS_VERSION = "1.13.8"
EKSCTL_VERSION = "0.5.0"
KSONNET_VERSION = "0.13.1"
KUBEFLOW_VERSION = "v0.4.1"
KUBETAIL_VERSION = "1.6.7"

EKS_NVIDIA_PLUGIN_VERSION = "1.12"

# https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html
EKS_AMI_ID = {"cpu": "ami-0d3998d69ebe9b214", "gpu": "ami-0484012ada3522476"}

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


def eks_setup(framework):
    """Function to download eksctl, kubectl, aws-iam-authenticator and ksonnet binaries
    Utilities:
    1. eksctl: create and manage cluster
    2. kubectl: create and manage runs on eks cluster
    3. aws-iam-authenticator: authenticate the instance to access eks with the appropriate aws credentials
    4. ksonnet: configure pod files and apply changes to the EKS cluster (will be deprecated soon, but no replacement available yet)
    :param framework: str
    """

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run_out = run(
        "eksctl version && kubectl version --short --client && aws-iam-authenticator version && ks version",
        warn=True,
    )

    eks_tools_installed = not run_out.return_code

    # Assume cluster with such a name is active
    eks_cluster_name = PR_EKS_CLUSTER_NAME_TEMPLATE.format(framework)

    if eks_tools_installed:
        eks_write_kubeconfig(eks_cluster_name, "us-west-2")
        return

    platform = run("uname -s").stdout.strip()

    eksctl_download_command = (
        f"curl --silent --location https://github.com/weaveworks/eksctl/releases/download/"
        f"{EKSCTL_VERSION}/eksctl_{platform}_amd64.tar.gz | tar xz -C /tmp"
    )

    kubectl_download_command = (
        f"curl --silent --location https://amazon-eks.s3-us-west-2.amazonaws.com/"
        f"{EKS_VERSION}/2019-08-14/bin/{platform.lower()}/amd64/kubectl -o /tmp/kubectl"
    )

    aws_iam_authenticator_download_command = (
        f"curl --silent --location https://amazon-eks.s3-us-west-2.amazonaws.com/"
        f"{EKS_VERSION}/2019-08-14/bin/{platform.lower()}/amd64/aws-iam-authenticator -o /tmp/aws-iam-authenticator"
    )

    ksonnet_download_command = (
        f"curl --silent --location https://github.com/ksonnet/ksonnet/releases/download/"
        f"v{KSONNET_VERSION}/ks_{KSONNET_VERSION}_{platform.lower()}_amd64.tar.gz -o /tmp/{KSONNET_VERSION}.tar.gz"
    )

    kubetail_download_command = (
        f"curl --silent --location https://raw.githubusercontent.com/johanhaleby/kubetail/"
        f"{KUBETAIL_VERSION}/kubetail -o /tmp/kubetail"
    )

    run(eksctl_download_command)
    run("mv /tmp/eksctl /usr/local/bin")

    run(kubectl_download_command)
    run("chmod +x /tmp/kubectl")
    run("mv /tmp/kubectl /usr/local/bin")

    run(aws_iam_authenticator_download_command)
    run("chmod +x /tmp/aws-iam-authenticator")
    run("mv /tmp/aws-iam-authenticator /usr/local/bin")

    run(ksonnet_download_command)
    run("tar -xf /tmp/{}.tar.gz -C /tmp --strip-components=1".format(KSONNET_VERSION))
    run("mv /tmp/ks /usr/local/bin")

    run(kubetail_download_command)
    run("chmod +x /tmp/kubetail")
    run("mv /tmp/kubetail /usr/local/bin")

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run("eksctl version")
    run("kubectl version --short --client")
    run("aws-iam-authenticator version")
    run("ks version")

    eks_write_kubeconfig(eks_cluster_name, "us-west-2")

    run(
        "kubectl apply -f https://raw.githubusercontent.com/NVIDIA"
        "/k8s-device-plugin/v{}/nvidia-device-plugin.yml".format(
            EKS_NVIDIA_PLUGIN_VERSION
        )
    )


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


def eks_multinode_cleanup(ctx, pod_name, job_name, namespace, env):
    """
    Function to cleanup resources created by EKS
    Use namespace as default if you do not create one.
    :param ctx:
    :param pod_name:
    :param job_name:
    :param namespace:
    :param env:
    :return:
    """
    # Operator specific cleanup
    if job_name == "openmpi-job":
        component, _ = pod_name.split("-master")
        ctx.run(f"ks component rm {component}", warn=True)
    else:
        ctx.run(f"ks delete {env} -c {job_name} -n {namespace}", warn=True)

    if "pytorch" not in namespace:
        ctx.run(f"ks delete {env}", warn=True)
    ctx.run(f"kubectl delete namespace {namespace}", warn=True)


def eks_multinode_get_logs(ctx, namespace, pod_name):
    """
    Function to get logs for a pod in the specified namespace.
    :param ctx:
    :param namespace:
    :param pod_name:
    :return:
    """
    return ctx.run(f"kubectl logs -n {namespace} -f {pod_name}").stdout


@retry(stop_max_attempt_number=120, wait_fixed=10000, retry_on_exception=retry_if_value_error)
def is_mpijob_launcher_pod_ready(ctx, namespace, job_name):
    """Check if the MpiJob Launcher Pod is Ready
    Args:
        ctx: Context
        namespace: str
        job_name: str
    """

    pod_name = ctx.run(
        f"kubectl get pods -n {namespace} -l mpi_job_name={job_name},mpi_role_type=launcher -o name"
    ).stdout.strip("\n")
    if pod_name:
        return pod_name
    else:
        raise ValueError("Launcher pod is not ready yet, try again.")


@retry(stop_max_attempt_number=40, wait_fixed=60000, retry_on_exception=retry_if_value_error)
def is_eks_multinode_training_complete(ctx, namespace, env, pod_name, job_name):
    """Function to check if the pod status has reached 'Completion' for multinode training.
    A separate method is required because kubectl commands for logs and status are different with namespaces.
    Args:
        namespace, pod_name, job_name: str
    """

    run_out = ctx.run(f"kubectl get pod -n {namespace} {pod_name} -o json")
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
                                 f"kubectl logs: {eks_multinode_get_logs(ctx, namespace, pod_name)}")
                    eks_multinode_cleanup(ctx, pod_name, job_name, namespace, env)
                    raise AttributeError("Container Error!")
            elif 'waiting' in container_status['state'] and \
                    container_status['state']['waiting']['reason'] == "CrashLoopBackOff":
                LOGGER.error(f"ERROR: The container run threw an error in waiting state. "
                             f"kubectl logs: {eks_multinode_get_logs(ctx, namespace, pod_name)}")
                eks_multinode_cleanup(ctx, pod_name, job_name, namespace, env)
                raise AttributeError("Error: CrashLoopBackOff!")
            elif 'waiting' in container_status['state'] or 'running' in container_status['state']:
                LOGGER.info("IN-PROGRESS: Container is either Creating or Running. Waiting to complete...")
                raise ValueError("IN-PROGRESS: Retry.")

    return False


def run_eks_multi_node_training_mpijob(namespace, app_name, custom_image, job_name, command_to_run, args_to_pass,
                                       path_to_ksonnet_app, cluster_size, eks_gpus_per_worker):
    """
    Function to run eks multinode training MPI job
    """
    kubeflow_version = "v0.5.1"
    pod_name = None
    env = f"{namespace}-env"
    ctx = Context()
    github_handler = GitHubHandler("aws", "kubeflow")
    github_token = github_handler.get_auth_token()

    ctx.run(f"kubectl create namespace {namespace}")

    if not os.path.exists(path_to_ksonnet_app):
        ctx.run(f"mkdir -p {path_to_ksonnet_app}")

    with ctx.cd(path_to_ksonnet_app):
        ctx.run(f"rm -rf {app_name}")
        ctx.run(f"ks init {app_name} --namespace {namespace}", env={"GITHUB_TOKEN": github_token})

        with ctx.cd(app_name):
            ctx.run(f"ks env add {env} --namespace {namespace}")
            # Check if the kubeflow registry exists and create. Registry will be available in each pod.
            registry_not_exist = ctx.run("ks registry list | grep kubeflow", warn=True)

            if registry_not_exist.return_code:
                ctx.run(
                    f"ks registry add kubeflow github.com/kubeflow/kubeflow/tree/{kubeflow_version}/kubeflow",
                    env={"GITHUB_TOKEN": github_token}
                )
                ctx.run(f"ks pkg install kubeflow/common@{kubeflow_version}", env={"GITHUB_TOKEN": github_token})
                ctx.run(f"ks pkg install kubeflow/mpi-job@{kubeflow_version}", env={"GITHUB_TOKEN": github_token})

            try:
                ctx.run("ks generate mpi-operator mpi-operator")
                # The latest mpi-operator docker image does not accept the gpus-per-node parameter
                # which is specified by the older spec file from v0.5.1.
                ctx.run("ks param set mpi-operator image mpioperator/mpi-operator:0.2.0")
                mpi_operator_start = ctx.run(f"ks apply {env} -c mpi-operator", warn=True)
                if mpi_operator_start.return_code:
                    raise RuntimeError(f"Failed to start mpi-operator:\n{mpi_operator_start.stderr}")

                LOGGER.info(
                    f"The mpi-operator package must be applied to {env} env before we can use mpiJob. "
                    f"Check status before moving on."
                )
                ctx.run("kubectl get crd")

                # Use Ksonnet to generate manifest files which are then applied to the default context.
                ctx.run(f"ks generate mpi-job-custom {job_name}")
                ctx.run(f"ks param set {job_name} replicas {cluster_size}")
                ctx.run(f"ks param set {job_name} gpusPerReplica {eks_gpus_per_worker}")
                ctx.run(f"ks param set {job_name} image {custom_image}")
                ctx.run(f"ks param set {job_name} command {command_to_run}")
                ctx.run(f"ks param set {job_name} args {args_to_pass}")

                # use `$ks show default` to see details.
                ctx.run(f"kubectl get pods -n {namespace} -o wide")
                LOGGER.info(f"Apply the generated manifest to the {env} env.")
                training_job_start = ctx.run(f"ks apply {env} -c {job_name}", warn=True)
                if training_job_start.return_code:
                    raise RuntimeError(f"Failed to start {job_name}:\n{training_job_start.stderr}")

                LOGGER.info("Check pods")
                ctx.run(f"kubectl get pods -n {namespace} -o wide")

                LOGGER.info(
                    "First the mpi-operator and the n-worker pods will be created and then "
                    "the launcher pod is created in the end. Use retries until launcher "
                    "pod's name is available to read logs."
                )
                complete_pod_name = is_mpijob_launcher_pod_ready(ctx, namespace, job_name)

                _, pod_name = complete_pod_name.split("/")
                LOGGER.info(f"The Pods have been created and the name of the launcher pod is {pod_name}")

                LOGGER.info(f"Wait for the {job_name} job to complete")
                if is_eks_multinode_training_complete(ctx, namespace, env, pod_name, job_name):
                    LOGGER.info(f"Wait for the {pod_name} pod to reach completion")
                    distributed_out = ctx.run(f"kubectl logs -n {namespace} -f {complete_pod_name}").stdout
                    LOGGER.info(distributed_out)
            finally:
                eks_multinode_cleanup(ctx, pod_name, job_name, namespace, env)
