import os
import random
import re

from datetime import datetime

import pytest

from invoke.context import Context

import test.test_utils.ec2 as ec2_utils
import test.test_utils.eks as eks_utils

from dlc.github_handler import GitHubHandler
from test.test_utils import is_pr_context, SKIP_PR_REASON


# Test only runs in region us-west-2, on instance type p3.16xlarge, on PR_EKS_CLUSTER_NAME_TEMPLATE cluster
@pytest.mark.skipif(is_pr_context(), reason=SKIP_PR_REASON)
@pytest.mark.integration("horovod")
@pytest.mark.model("resnet")
@pytest.mark.multinode("multinode")
def test_eks_tensorflow_multi_node_training_gpu(tensorflow_training, example_only):
    eks_cluster_size = 3
    ec2_instance_type = "p3.16xlarge"

    eks_gpus_per_worker = ec2_utils.get_instance_num_gpus(instance_type=ec2_instance_type)

    _run_eks_tensorflow_multinode_training_resnet50_mpijob(tensorflow_training, eks_cluster_size, eks_gpus_per_worker)


def _run_eks_tensorflow_multinode_training_resnet50_mpijob(example_image_uri, cluster_size, eks_gpus_per_worker):
    """
    Run Tensorflow distributed training on EKS using horovod docker images with synthetic dataset
    :param example_image_uri:
    :param cluster_size:
    :param eks_gpus_per_worker:
    :return: None
    """
    user = Context().run("echo $USER").stdout.strip("\n")
    framework_version = re.search(r"\d+(\.\d+)+", example_image_uri).group()
    major_version = framework_version.split(".")[0]
    random.seed(f"{example_image_uri}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_tag = f"{user}-{random.randint(1, 10000)}"
    namespace = f"tf{major_version}-multi-node-train-{'py2' if 'py2' in example_image_uri else 'py3'}-{unique_tag}"
    job_name = f"tf-resnet50-horovod-job-{unique_tag}"

    script_name = ("/deep-learning-models/models/resnet/tensorflow2/train_tf2_resnet.py" if major_version == "2" else
                   "/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py")

    command_to_run = ("mpirun,-mca,btl_tcp_if_exclude,lo,-mca,pml,ob1,-mca,btl,^openib,--bind-to,none,-map-by,slot,"
                      "-x,LD_LIBRARY_PATH,-x,PATH,-x,NCCL_SOCKET_IFNAME=eth0,-x,NCCL_DEBUG=INFO,python,") + script_name
    args_to_pass = ("-- --synthetic,--batch_size,128,--num_batches,100,--clear_log,2" if major_version == "2" else
                    "-- --num_epochs=1,--synthetic")

    home_dir = Context().run("echo $HOME").stdout.strip("\n")
    path_to_ksonnet_app = os.path.join(home_dir, f"tensorflow_multi_node_eks_test-{unique_tag}")
    app_name = f"kubeflow-tf-hvd-mpijob-{unique_tag}"

    _run_eks_tensorflow_multi_node_training_mpijob(namespace, app_name, example_image_uri, job_name,
                                                   command_to_run, args_to_pass, path_to_ksonnet_app,
                                                   cluster_size, eks_gpus_per_worker)


def _run_eks_tensorflow_multi_node_training_mpijob(namespace, app_name, custom_image, job_name,
                                                   command_to_run, args_to_pass, path_to_ksonnet_app,
                                                   cluster_size, eks_gpus_per_worker):
    """
    Run Tensorflow distributed training on EKS using horovod docker images using MPIJob
    :param namespace:
    :param app_name:
    :param custom_image:
    :param job_name:
    :param command_to_run:
    :param args_to_pass:
    :param path_to_ksonnet_app:
    :param cluster_size:
    :param eks_gpus_per_worker:
    :return: None
    """
    KUBEFLOW_VERSION = "v0.5.1"
    pod_name = None
    env = f"{namespace}-env"
    ctx = Context()
    github_handler = GitHubHandler("aws", "kubeflow")
    github_handler.set_ksonnet_env()

    ctx.run(f"kubectl create namespace {namespace}")

    if not os.path.exists(path_to_ksonnet_app):
        ctx.run(f"mkdir -p {path_to_ksonnet_app}")

    with ctx.cd(path_to_ksonnet_app):
        ctx.run(f"rm -rf {app_name}")
        ctx.run(f"ks init {app_name} --namespace {namespace}")

        with ctx.cd(app_name):
            ctx.run(f"ks env add {env} --namespace {namespace}")
            # Check if the kubeflow registry exists and create. Registry will be available in each pod.
            registry_not_exist = ctx.run("ks registry list | grep kubeflow", warn=True)

            if registry_not_exist.return_code:
                ctx.run(
                    f"ks registry add kubeflow github.com/kubeflow/kubeflow/tree/{KUBEFLOW_VERSION}/kubeflow",
                )
                ctx.run(f"ks pkg install kubeflow/common@{KUBEFLOW_VERSION}")
                ctx.run(f"ks pkg install kubeflow/mpi-job@{KUBEFLOW_VERSION}")

            try:
                ctx.run("ks generate mpi-operator mpi-operator")
                # The latest mpi-operator docker image does not accept the gpus-per-node parameter
                # which is specified by the older spec file from v0.5.1.
                ctx.run("ks param set mpi-operator image mpioperator/mpi-operator:0.2.0")
                ctx.run("ks param set mpi-operator kubectlDeliveryImage mpioperator/kubectl-delivery:0.2.0")
                mpi_operator_start = ctx.run(f"ks apply {env} -c mpi-operator", warn=True)
                if mpi_operator_start.return_code:
                    raise RuntimeError(f"Failed to start mpi-operator:\n{mpi_operator_start.stderr}")

                eks_utils.LOGGER.info(
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
                eks_utils.LOGGER.info(f"Apply the generated manifest to the {env} env.")
                training_job_start = ctx.run(f"ks apply {env} -c {job_name}", warn=True)
                if training_job_start.return_code:
                    raise RuntimeError(f"Failed to start {job_name}:\n{training_job_start.stderr}")

                eks_utils.LOGGER.info("Check pods")
                ctx.run(f"kubectl get pods -n {namespace} -o wide")

                eks_utils.LOGGER.info(
                    "First the mpi-operator and the n-worker pods will be created and then "
                    "the launcher pod is created in the end. Use retries until launcher "
                    "pod's name is available to read logs."
                )
                complete_pod_name = eks_utils.is_mpijob_launcher_pod_ready(ctx, namespace, job_name)

                _, pod_name = complete_pod_name.split("/")
                eks_utils.LOGGER.info(f"The Pods have been created and the name of the launcher pod is {pod_name}")

                eks_utils.LOGGER.info(f"Wait for the {job_name} job to complete")
                if eks_utils.is_eks_multinode_training_complete(ctx, namespace, env, pod_name, job_name):
                    eks_utils.LOGGER.info(f"Wait for the {pod_name} pod to reach completion")
                    distributed_out = ctx.run(f"kubectl logs -n {namespace} -f {complete_pod_name}").stdout
                    eks_utils.LOGGER.info(distributed_out)
            finally:
                eks_utils.eks_multinode_cleanup(ctx, pod_name, job_name, namespace, env)
