"""
Helper functions for EKS Integration Tests
"""

import os
import sys
import json
import logging

from retrying import retry
from invoke import run
from invoke.context import Context

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


def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)


@retry(
    stop_max_attempt_number=240,
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

    run_out = run(eksctl_check_cluster_command, warn_only=True)

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
    eksctl_write_kubeconfig_command = """eksctl utils write-kubeconfig \
                                         --name {} --region {}""".format(
        eks_cluster_name, region
    )
    run(eksctl_write_kubeconfig_command)

    # run(f"aws eks --region us-west-2 update-kubeconfig --name {eks_cluster_name} --kubeconfig /root/.kube/config --role-arn arn:aws:iam::669063966089:role/nikhilsk-eks-test-role")

    run("cat /root/.kube/config", warn=True)


def run_eks_mxnet_multi_node_training(namespace, app_name, job_name, remote_yaml_file_path):
    """Run MXNet distributed training on EKS using MXNet Operator
    Args:
    namespace, app_name, job_name, remote_yaml_file_path
    """

    training_result = False

    # Namespaces will allow parallel runs on the same cluster. Create namespace if it doesnt exist.
    does_namespace_exist = run("kubectl get namespace | grep {}".format(namespace),
                               warn_only=True)
    if not does_namespace_exist:
        run("kubectl create namespace {}".format(namespace))

    # Create a new ksonnet app.
    run("rm -rf {}".format(app_name))

    context = Context()

    #with hide('running'):
    #    _, github_token = utils.get_github_token()
    #    with shell_env(GITHUB_TOKEN=github_token):
    run("ks init {}".format(app_name))

    with context.cd(f"{app_name}"):
        context.run("ks env set default --namespace {}".format(namespace))

        # Check if the kubeflow registry exists and create. Registry will be available in each pod.
        does_registry_exist = run("ks registry list | grep kubeflow", warn_only=True)
        if not does_registry_exist:
            #with hide('running'):
            #    _, github_token = utils.get_github_token()
            #    with shell_env(GITHUB_TOKEN=github_token):
            run("ks registry add kubeflow github.com/kubeflow/kubeflow/tree/{}/kubeflow".format(KUBEFLOW_VERSION))
            run("ks pkg install kubeflow/mxnet-job@{}".format(KUBEFLOW_VERSION))

            run("ks generate mxnet-operator mxnet-operator")

            try:
                # use `$ks show default` to see details.
                run("ks apply default -c mxnet-operator")
                # Delete old job with same name if exists
                run("kubectl delete -f {}".format(remote_yaml_file_path), warn_only=True)
                run("kubectl create -f {}".format(remote_yaml_file_path))
                if is_mxnet_eks_multinode_training_complete(job_name, remote_yaml_file_path):
                    training_result = True
            except Exception as e:
                raise Exception("something went wrong! Exception - {}".format(e))
            finally:
                run("kubectl delete -f {}".format(remote_yaml_file_path), warn_only=True)
                # If different versions of kubeflow used in the cluster, crd must be deleted.
                run("kubectl delete crd mxjobs.kubeflow.org")
                eks_multinode_cleanup("", job_name, namespace)

    return training_result


@retry(stop_max_attempt_number=40, wait_fixed=6000, retry_on_exception=retry_if_value_error)
def is_mxnet_eks_multinode_training_complete(job_name, remote_yaml_file_path):
    """Function to check job and pod status for multinode training.
    A separate method is required because kubectl commands for logs and status are different with namespaces.
    Args:
        job_name: str, remote_yaml_file_path: str
    """
    run_out = run("kubectl get mxjobs {} -o json".format(job_name))
    pod_info = json.loads(run_out.stdout)

    if 'status' not in pod_info:
        raise ValueError("Waiting for job to launch...")

    # Job_phase can be one of the Creating, Running, Cleanup, Failed, Done
    # Job state can be one of the Running, Succeeded, Failed
    if 'phase' in pod_info['status']:
        job_phase = pod_info['status']['phase']
        job_state = pod_info['status']['state']
        LOGGER.info("Current job phase: %s", job_phase)

        if 'Failed' in job_state:
                LOGGER.info("Failure: Job failed to run and the pods are getting terminated.")
        elif 'Succeeded' in job_state:
              if 'Done' in job_phase or 'CleanUp' in job_phase:
                  LOGGER.info("SUCCESS: Job is complete. Pods are getting terminated.")
                  return True
        elif 'Running' in job_state:
            if 'Creating' in job_phase:
                LOGGER.info("IN-PROGRESS: Container is either Creating. Waiting to complete...")
                raise ValueError("IN-PROGRESS: Container getting created.")
            elif 'Running' in job_phase:
                # Print logs generated
                run("kubetail $(kubectl get pods | grep {} | cut -f 1 -d ' ' | paste -s -d, -) --follow "
                    "false".format(job_name +"-worker"), warn_only=True)
                raise ValueError("IN-PROGRESS: Job is running.")
            elif 'CleanUp' in job_phase or 'Failed' in job_phase:
                LOGGER.info("Failed: The job failed to execute. Pods are getting terminated.")
            elif 'Done' in job_phase:
                LOGGER.info("Failed: The job failed to execute. Pods are getting terminated.")

    return False


def eks_multinode_cleanup(pod_name, job_name, namespace):
    """Function to cleanup resources created by EKS
    Use namespace as default if you do not create one.
    Args:
        pod_name, job_name, namespace: str
    """

    # Operator specific cleanup
    if job_name == "openmpi-job":
        component,_ = pod_name.split("-master")
        run("ks component rm {}".format(component), warn_only=True)
    else:
        run("ks delete default -c {}".format(job_name), warn_only=True)

    run("ks delete default", warn_only=True)
    run("kubectl delete namespace {}".format(namespace), warn_only=True)


def generate_mxnet_multinode_yaml_file(container_image, job_name, num_workers, num_servers, gpu_limit, command, args, remote_yaml_file_path):
    """Function that writes the yaml file for a given container_image and commands to create a pod.
    Args:
        container_img, job_name, num_workers, num_servers, gpu_limit, command, args: list, remote_yaml_file_path: str
    """

    yaml_data = {
        "apiVersion": "kubeflow.org/v1alpha1",
        "kind": "MXJob",
        "metadata": {
            "name": job_name
        },
        "spec": {
            "jobMode": "dist",
            "replicaSpecs": [
                {
                    "replicas": 1,
                    "mxReplicaType": "SCHEDULER",
                    "PsRootPort": 9000,
                    "template": {
                        "spec": {
                            "containers":[
                                {
                                    "name": "mxnet",
                                    "image": container_image
                                }],
                            "restartPolicy": "OnFailure",
                        }
                    }
                },

                {
                    "replicas": num_servers,
                    "mxReplicaType": "SERVER",
                    "template": {
                        "spec": {
                            "containers":[
                                {
                                    "name": "mxnet",
                                    "image": container_image
                                }],
                        }
                    }
                },

                {
                    "replicas": num_workers,
                    "mxReplicaType": "WORKER",
                    "template": {
                        "spec": {
                            "containers":[
                                {
                                   "name": "mxnet",
                                   "image": container_image,
                                   "command": command,
                                   "args": args,
                                   "resources": {
                                       "limits": {
                                           "nvidia.com/gpu": gpu_limit
                                        }
                                   }
                                }],
                          "restartPolicy": "OnFailure",
                        }
                    }
                }
            ]
        }
    }
    put(StringIO(unicode(yaml.dump(yaml_data, default_flow_style=False))), remote_yaml_file_path)

    LOGGER.info("Uploaded generated yaml file to %s", remote_yaml_file_path)
