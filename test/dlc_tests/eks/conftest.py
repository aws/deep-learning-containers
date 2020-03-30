import time
import logging
import pytest
import sys
from invoke import run, sudo

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

import test.test_utils.eks as eks_utils

EKS_VERSION = "1.13.8"
EKSCTL_VERSION = "0.5.0"
KSONNET_VERSION = "0.13.1"
KUBEFLOW_VERSION = "v0.4.1"
EKS_NVIDIA_PLUGIN_VERSION = "1.12"
KUBETAIL_VERSION = "1.6.7"

EKS_NVIDIA_PLUGIN_VERSION = "1.12"
# https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html
EKS_AMI_ID = {"cpu": "ami-0d3998d69ebe9b214", "gpu": "ami-0484012ada3522476"}

SSH_PUBLIC_KEY_NAME = "dlc-ec2-keypair-prod"
PR_EKS_CLUSTER_NAME = "dlc-pr-eks-test-cluster"


@pytest.fixture(scope="session")
def eks_setup():
    """Function to download eksctl, kubectl, aws-iam-authenticator and ksonnet binaries
    Utilities:
    1. eksctl: create and manage cluster
    2. kubectl: create and manage runs on eks cluster
    3. aws-iam-authenticator: authenticate the instance to access eks with the appropriate aws credentials
    4. ksonnet: configure pod files and apply changes to the EKS cluster (will be deprecated soon, but no replacement available yet)
    """
    # Run a quick check that the binaries are available in the PATH by listing the 'version'

    eks_tools_installed = True

    run_out = run(
        "eksctl version && kubectl version --short --client && aws-iam-authenticator version && ks version",
        warn=True,
    )

    eks_tools_installed = not run_out.return_code

    if eks_tools_installed:
        eks_utils.eks_write_kubeconfig(PR_EKS_CLUSTER_NAME, "us-west-2")
        return

    eksctl_download_command = """curl --silent --location \
    https://github.com/weaveworks/eksctl/releases/download/{}/eksctl_Darwin_amd64.tar.gz | \
    tar xz -C /tmp""".format(
        EKSCTL_VERSION
    )

    kubectl_download_command = """curl --silent --location \
    https://amazon-eks.s3-us-west-2.amazonaws.com/{}/2019-08-14/bin/darwin/amd64/kubectl \
        -o /tmp/kubectl""".format(
        EKS_VERSION
    )

    aws_iam_authenticator_download_command = """curl --silent --location \
    https://amazon-eks.s3-us-west-2.amazonaws.com/{}/2019-08-14/bin/darwin/amd64/aws-iam-authenticator \
        -o /tmp/aws-iam-authenticator""".format(
        EKS_VERSION
    )

    # TODO: change 'linux' to 'darwin' for MacOS
    ksonnet_download_command = """wget https://github.com/ksonnet/ksonnet/releases/download/v{0}/ks_{0}_darwin_amd64.tar.gz \
        -O /tmp/{0}.tar.gz """.format(
        KSONNET_VERSION
    )

    kubetail_download_command = """curl --silent --location \
        https://raw.githubusercontent.com/johanhaleby/kubetail/{}/kubetail \
        -o /tmp/kubetail""".format(
        KUBETAIL_VERSION
    )

    run(eksctl_download_command)
    sudo("mv /tmp/eksctl /usr/local/bin")

    run(kubectl_download_command)
    run("chmod +x /tmp/kubectl")
    sudo("mv /tmp/kubectl /usr/local/bin")

    run(aws_iam_authenticator_download_command)
    run("chmod +x /tmp/aws-iam-authenticator")
    sudo("mv /tmp/aws-iam-authenticator /usr/local/bin")

    run(ksonnet_download_command)
    run("tar -xf /tmp/{}.tar.gz -C /tmp --strip-components=1".format(KSONNET_VERSION))
    sudo("mv /tmp/ks /usr/local/bin")

    run(kubetail_download_command)
    run("chmod +x /tmp/kubetail")
    sudo("mv /tmp/kubetail /usr/local/bin")

    # Run a quick check that the binaries are available in the PATH by listing the 'version'
    run("eksctl version")
    run("kubectl version --short --client")
    run("aws-iam-authenticator version")
    run("ks version")

    eks_utils.eks_write_kubeconfig(PR_EKS_CLUSTER_NAME, "us-west-2")


@pytest.fixture(scope="session")
def setup_eks_cluster(processor_type, instance_type, num_nodes, eks_cluster_name):
    """Function to start, setup and verify the status of EKS cluster. It verifies if the cluster
    exists, if not, creates one. If a cluster exists and is 'ACTIVE', it verifies if the cluster
    nodegroup exists and attaches the nodegroup required. For the purpose of this test, the EKS
    cluster is associated with only one nodegroup.
    Args:
        processor_type, instance_type, num_nodes: str
    """

    if not eks_utils.is_eks_cluster_active(eks_cluster_name):
        LOGGER.info(
            "No associated nodegroup found for cluster: %s. Creating nodegroup.",
            eks_cluster_name,
        )
    else:
        LOGGER.info(
            "No active cluster named %s found. Creating the cluster.", eks_cluster_name
        )
        eks_utils.create_eks_cluster(
            eks_cluster_name,
            processor_type,
            num_nodes,
            instance_type,
            SSH_PUBLIC_KEY_NAME,
        )

    eks_utils.eks_write_kubeconfig(eks_cluster_name)

    run("kubectl delete all --all", warn_only=True)

    if processor_type == "gpu":
        run(
            "kubectl apply -f https://raw.githubusercontent.com/NVIDIA"
            "/k8s-device-plugin/v{}/nvidia-device-plugin.yml".format(
                EKS_NVIDIA_PLUGIN_VERSION
            )
        )

    LOGGER.info(
        "Cluster is active and associated nodegroup configured. "
        "Kubeconfig has been updated. EKS setup complete."
    )
