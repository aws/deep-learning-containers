import os
import sys
import logging

from multiprocessing import Pool

import pytest

from invoke import run
from invoke.context import Context

import test_utils.eks as eks_utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

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
PR_EKS_CLUSTER_NAME = "dlc-eks-pr-{}-test-cluster"

def assign_sagemaker_instance_type(image):
    if "tensorflow" in image:
        return "ml.p3.8xlarge" if "gpu" in image else "ml.c4.4xlarge"
    else:
        return "ml.p2.8xlarge" if "gpu" in image else "ml.c4.8xlarge"


def generate_sagemaker_pytest_cmd(image):
    """
    Parses the image ECR url and returns appropriate pytest command

    :param image: ECR url of image
    :return: <tuple> pytest command to be run, path where it should be executed, image tag
    """
    region = os.getenv("AWS_REGION", "us-west-2")
    integration_path = os.path.join("integration", "sagemaker")
    account_id = os.getenv("ACCOUNT_ID", image.split(".")[0])
    docker_base_name, tag = image.split("/")[1].split(":")

    # Assign instance type
    instance_type = assign_sagemaker_instance_type(image)

    # Get path to test directory
    find_path = docker_base_name.split("-")

    # NOTE: We are relying on the fact that repos are defined as <context>-<framework>-<job_type> in our infrastructure
    framework = find_path[1]
    job_type = find_path[2]
    path = os.path.join("test", "sagemaker_tests", framework, job_type)
    aws_id_arg = "--aws-id"
    docker_base_arg = "--docker-base-name"
    instance_type_arg = "--instance-type"

    # Conditions for modifying tensorflow SageMaker pytest commands
    if framework == "tensorflow":
        if job_type == "training":
            aws_id_arg = "--account-id"

            # NOTE: We are relying on tag structure to get TF major version. If tagging changes, this will break.
            tf_major_version = tag.split("-")[-1].split(".")[0]
            path = os.path.join(
                "sagemaker_tests", framework, f"{framework}{tf_major_version}_training"
            )
        else:
            aws_id_arg = "--registry"
            docker_base_arg = "--repo"
            integration_path = os.path.join(integration_path, "test_tfs.py")
            instance_type_arg = "--instance-types"

    test_report = os.path.join(os.getcwd(), "test", f"{tag}.xml")
    return (
        f"pytest {integration_path} --region {region} {docker_base_arg} "
        f"{docker_base_name} --tag {tag} {aws_id_arg} {account_id} {instance_type_arg} {instance_type} "
        f"--junitxml {test_report}",
        path,
        tag,
    )


def run_sagemaker_pytest_cmd(image):
    """
    Run pytest in a virtual env for a particular image

    Expected to run via multiprocessing

    :param image: ECR url
    """

    pytest_command, path, tag = generate_sagemaker_pytest_cmd(image)

    context = Context()
    with context.cd(path):
        context.run(f"virtualenv {tag}")
        with context.prefix(f"source {tag}/bin/activate"):
            context.run("pip install -r requirements.txt", warn=True)
            context.run(pytest_command)


def run_sagemaker_tests(images):
    """
    Function to set up multiprocessing for SageMaker tests

    :param images:
    """
    pool_number = len(images)
    with Pool(pool_number) as p:
        p.map(run_sagemaker_pytest_cmd, images)


def pull_dlc_images(images):
    """
    Pulls DLC images to CodeBuild jobs before running PyTest commands
    """
    for image in images:
        run(f"docker pull {image}", hide='out')

def eks_setup(framework):
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
    https://github.com/weaveworks/eksctl/releases/download/{}/eksctl_Linux_amd64.tar.gz | \
    tar xz -C /tmp""".format(
        EKSCTL_VERSION
    )

    kubectl_download_command = """curl --silent --location \
    https://amazon-eks.s3-us-west-2.amazonaws.com/{}/2019-08-14/bin/linux/amd64/kubectl \
        -o /tmp/kubectl""".format(
        EKS_VERSION
    )

    aws_iam_authenticator_download_command = """curl --silent --location \
    https://amazon-eks.s3-us-west-2.amazonaws.com/{}/2019-08-14/bin/linux/amd64/aws-iam-authenticator \
        -o /tmp/aws-iam-authenticator""".format(
        EKS_VERSION
    )

    # TODO: change 'linux' to 'darwin' for MacOS
    ksonnet_download_command = """curl --silent --location https://github.com/ksonnet/ksonnet/releases/download/v{0}/ks_{0}_linux_amd64.tar.gz \
        -o /tmp/{0}.tar.gz""".format(
        KSONNET_VERSION
    )

    kubetail_download_command = """curl --silent --location \
        https://raw.githubusercontent.com/johanhaleby/kubetail/{}/kubetail \
        -o /tmp/kubetail""".format(
        KUBETAIL_VERSION
    )


    LOGGER.info("aws sts get-caller-identity")
    run_out = run("aws sts get-caller-identity")
    LOGGER.info(run_out)
    run(eksctl_download_command)
    LOGGER.info("********* whoami")
    run_out = run("whoami")
    LOGGER.info(run_out.stdout)
    LOGGER.info("********* Listing /tmp")
    run_out = run("ls -l /tmp")
    LOGGER.info(run_out.stdout)
    LOGGER.info("********* Listing /usr/local/bin")
    run_out = run("ls -l /usr/local/bin")
    LOGGER.info(run_out.stdout)
    run_out = run("mv /tmp/eksctl /usr/local/bin")
    LOGGER.info(run_out.stdout)

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

    # # Create the cluster if it doesn't exist:
    # if not eks_utils.is_eks_cluster_active(PR_EKS_CLUSTER_NAME):
    #     eks_utils.create_eks_cluster(PR_EKS_CLUSTER_NAME, "gpu", "3", "p3.16xlarge", "dlc-ec2-keypair-prod", region="us-west-2")
    #     #run(f"eksctl create cluster dlc-{PR_EKS_CLUSTER_NAME} --nodes 3 --node-type=p3.16xlarge --timeout=40m --ssh-access --ssh-public-key dlc-ec2-keypair-prod --region us-east-1 --auto-kubeconfig --region us-west-2")

    eks_cluster_name = PR_EKS_CLUSTER_NAME.format(framework)
    eks_utils.eks_write_kubeconfig(, "us-west-2")

    run("kubectl apply -f https://raw.githubusercontent.com/NVIDIA"
        "/k8s-device-plugin/v{}/nvidia-device-plugin.yml".format(EKS_NVIDIA_PLUGIN_VERSION))

def main():
    # Define constants
    test_type = os.getenv("TEST_TYPE")
    dlc_images = os.getenv("DLC_IMAGES")

    # dlc_images = '669063966089.dkr.ecr.us-west-2.amazonaws.com/pr-mxnet-training:training-gpu-py3 669063966089.dkr.ecr.us-west-2.amazonaws.com/pr-tensorflow-training:training-gpu-py3-2.1.0 ' \
    #              '669063966089.dkr.ecr.us-west-2.amazonaws.com/pr-tensorflow-training:training-gpu-py3-1.15.2 669063966089.dkr.ecr.us-west-2.amazonaws.com/pr-pytorch-training:training-gpu-py3'

    if test_type in ("sanity", "ecs", "ec2", "eks"):
        report = os.path.join(os.getcwd(), "test", f"{test_type}.xml")

        # PyTest must be run in this directory to avoid conflicting w/ sagemaker_tests conftests
        os.chdir(os.path.join("test", "dlc_tests"))

        # Pull images for necessary tests
        if test_type == "sanity":
            pull_dlc_images(dlc_images.split(" "))
        if test_type == "eks":
            framework = "mxnet" if "mxnet" in dlc_images else "pytorch" if "pytorch" in dlc_images else "tensorflow"
            eks_setup(framework)

        # Execute dlc_tests pytest command
        pytest_cmd = ["-s", test_type, f"--junitxml={report}", "-n=auto"]
        sys.exit(pytest.main(pytest_cmd))
    elif test_type == "sagemaker":
        run_sagemaker_tests(dlc_images.split(" "))
    else:
        raise NotImplementedError("Tests only support sagemaker and sanity currently")


if __name__ == "__main__":
    main()
