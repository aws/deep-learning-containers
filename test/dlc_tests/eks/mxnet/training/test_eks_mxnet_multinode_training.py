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
import test.test_utils.ec2 as ec2_utils
from test.test_utils import is_pr_context, SKIP_PR_REASON

LOGGER = eks_utils.LOGGER


@pytest.mark.skipif(is_pr_context(), reason=SKIP_PR_REASON)
@pytest.mark.multinode("multinode(3)")
def test_eks_mxnet_multi_node_training_horovod_mnist(mxnet_training, example_only):
    """Run MXNet distributed training on EKS using docker images with MNIST dataset"""

    eks_cluster_size = "3"
    ec2_instance_type = "p3.16xlarge"

    eks_gpus_per_worker = ec2_utils.get_instance_num_gpus(instance_type=ec2_instance_type)
    _run_eks_mxnet_multinode_training_horovod_mpijob(mxnet_training, eks_cluster_size, eks_gpus_per_worker)


def _run_eks_mxnet_multinode_training_horovod_mpijob(example_image_uri, cluster_size, eks_gpus_per_worker):
    
    LOGGER.info("Starting run_eks_mxnet_multi_node_training on MNIST dataset using horovod")
    LOGGER.info("The test will run on an example image %s", example_image_uri)

    user = Context().run("echo $USER").stdout.strip("\n")
    random.seed(f"{example_image_uri}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_tag = f"{user}-{random.randint(1, 10000)}"

    namespace = f"mx-multi-node-train-{'py2' if 'py2' in example_image_uri else 'py3'}-{unique_tag}"
    job_name = f"mxnet-mnist-horovod-job-{unique_tag}"

    LOGGER.debug(f"Namespace: {namespace}")

    local_template_file_path = os.path.join(
        "eks",
        "eks_manifest_templates",
        "mxnet",
        "training",
        "multi_node_training_horovod_mnist.yaml"
    )

    remote_yaml_file_path = os.path.join(os.sep, "tmp", f"tensorflow_multi_node_training_{unique_tag}.yaml")

    replace_dict = {
        "<JOB_NAME>": job_name,
        "<NUM_WORKERS>": cluster_size,
        "<CONTAINER_IMAGE>": example_image_uri,
        "<GPUS>": str(eks_gpus_per_worker)
    }

    eks_utils.write_eks_yaml_file_from_template(local_template_file_path, remote_yaml_file_path, replace_dict)

    _run_eks_multi_node_training_mpijob(namespace, job_name, remote_yaml_file_path)


@pytest.mark.skipif(is_pr_context(), reason=SKIP_PR_REASON)
@pytest.mark.multinode("multinode(3)")
def test_eks_mxnet_multinode_training(mxnet_training, example_only):
    """Run MXNet distributed training on EKS using docker images with MNIST dataset"""
    random.seed(f"{mxnet_training}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 6000)
    namespace = f"mxnet-multi-node-training-{unique_id}"
    job_name = f"kubeflow-mxnet-gpu-dist-job-{unique_id}"

    # TODO: This should either be dynamic or at least global variables
    num_workers = "3"
    num_servers = "2"
    gpu_limit = "1"
    epochs = '"20"'
    layers = '"2"'
    gpus = '"0"'

    local_template_file_path = os.path.join(
        "eks",
        "eks_manifest_templates",
        "mxnet",
        "training",
        "multi_node_gpu_training.yaml"
    )

    remote_yaml_file_path = os.path.join(os.sep, "tmp", f"mxnet_multi_node_training_{unique_id}.yaml")

    replace_dict = {
        "<JOB_NAME>": job_name,
        "<NUM_SERVERS>": num_servers,
        "<NUM_WORKERS>": num_workers,
        "<CONTAINER_IMAGE>": mxnet_training,
        "<EPOCHS>": epochs,
        "<LAYERS>": layers,
        "<GPUS>": gpus,
        "<GPU_LIMIT>": gpu_limit
    }

    eks_utils.write_eks_yaml_file_from_template(local_template_file_path, remote_yaml_file_path, replace_dict)

    training_result = _run_eks_mxnet_multi_node_training(namespace, job_name, remote_yaml_file_path)
    assert training_result, "EKS multinode training failed"


def _run_eks_mxnet_multi_node_training(namespace, job_name, remote_yaml_file_path):
    """Run MXNet distributed training on EKS using MXNet Operator
    Args:
    namespace, job_name, remote_yaml_file_path
    """

    training_result = False

    # Namespaces will allow parallel runs on the same cluster. Create namespace if it doesnt exist.
    does_namespace_exist = run(f"kubectl get namespace | grep {namespace}", warn=True)
    if not does_namespace_exist:
        run(f"kubectl create namespace {namespace}")

    try:
        
        # Delete old job with same name if exists
        run(f"kubectl delete -f {remote_yaml_file_path}", warn=True)
        run(f"kubectl create -f {remote_yaml_file_path} -n {namespace}")
        if is_mxnet_eks_multinode_training_complete(job_name, namespace):
            training_result = True
    finally:
        eks_utils.eks_multinode_cleanup(remote_yaml_file_path, namespace)

    return training_result


@retry(stop_max_attempt_number=60, wait_fixed=12000, retry_on_exception=eks_utils.retry_if_value_error)
def is_mxnet_eks_multinode_training_complete(job_name, namespace):
    """Function to check job and pod status for multinode training.
    A separate method is required because kubectl commands for logs and status are different with namespaces.
    Args:
        job_name: str, remote_yaml_file_path: str
    """
    run_out = run(f"kubectl get mxjobs -n {namespace} {job_name} -o json", warn=True)
    if run_out.stdout is not None or run_out.stdout != "":
        job_info = json.loads(run_out.stdout)
        LOGGER.debug(f"Job info: {job_info}")

    if 'status' not in job_info:
        raise ValueError("Waiting for job to launch...")
    job_status = job_info['status']
    if 'conditions' not in job_status:
        raise ValueError("Waiting for job to launch...")
    job_conditions = job_status['conditions']
    if len(job_conditions) == 0:
        raise ValueError("Waiting for job to launch...")
    else:
        # job_conditions at least with length 1
        if 'status' in job_conditions[0]:
            job_created = job_conditions[0]['status']
            if 'message' in job_conditions[0] and len(job_conditions) == 1:
                LOGGER.info(job_conditions[0]['message'])
            if not job_created:
                raise ValueError("Waiting for job to be created...")
            if len(job_conditions) == 1:
                raise ValueError("Waiting for job to run...")
            # job_conditions at least with length 2
            if 'status' in job_conditions[1]:
                job_running = job_conditions[1]['status']
                if 'message' in job_conditions[1] and len(job_conditions) == 2:
                    LOGGER.info(job_conditions[1]['message'])
                if not job_running:
                    raise ValueError("Waiting for job to run...")
                if len(job_conditions) == 2:
                    raise ValueError("Waiting for job to complete...")
                # job_conditions at least with length 3
                if 'status' in job_conditions[2]:
                    job_succeed = job_conditions[2]['status']
                    if 'message' in job_conditions[2]:
                        LOGGER.info(job_conditions[2]['message'])
                    if not job_succeed:
                        if job_running:
                            raise ValueError("Waiting for job to complete...")
                        else:
                            return False
                    return True
            else:
                raise ValueError("Waiting for job to run...")
        else:
            raise ValueError("Waiting for job to launch...")
    return False


def _run_eks_multi_node_training_mpijob(namespace, job_name, remote_yaml_file_path):
    """
    Function to run eks multinode training MPI job
    """

    run(f"kubectl create namespace {namespace}")

    try:
        training_job_start = run(f"kubectl create -f {remote_yaml_file_path} -n {namespace}", warn=True)
        if training_job_start.return_code:
            raise RuntimeError(f"Failed to start {job_name}:\n{training_job_start.stderr}")

        LOGGER.info("Check pods")
        run(f"kubectl get pods -n {namespace} -o wide")

        complete_pod_name = eks_utils.is_mpijob_launcher_pod_ready(namespace, job_name)

        _, pod_name = complete_pod_name.split("/")
        LOGGER.info(f"The Pods have been created and the name of the launcher pod is {pod_name}")

        LOGGER.info(f"Wait for the {job_name} job to complete")
        if eks_utils.is_eks_multinode_training_complete(remote_yaml_file_path, namespace, pod_name, job_name):
            LOGGER.info(f"Wait for the {pod_name} pod to reach completion")
            distributed_out = run(f"kubectl logs -n {namespace} -f {complete_pod_name}").stdout
            LOGGER.info(distributed_out)
    finally:
        eks_utils.eks_multinode_cleanup(remote_yaml_file_path, namespace)
