import json
import os
import random
import datetime

import pytest
from invoke import run
from invoke.context import Context
from retrying import retry

import test.test_utils.eks as eks_utils
from test.test_utils import is_pr_context, SKIP_PR_REASON


LOGGER = eks_utils.LOGGER


@pytest.mark.skipif(not is_pr_context(), reason="Skip this test. It is already tested under PR context and we do not have enough resouces to test it again on mainline pipeline")
def test_eks_pytorch_single_node_training(pytorch_training):
    """
    Function to create a pod using kubectl and given container image, and run MXNet training
    Args:
        :param setup_utils: environment in which EKS tools are setup
        :param pytorch_training: the ECR URI
    """

    training_result = False

    rand_int = random.randint(4001, 6000)

    yaml_path = os.path.join(os.sep, "tmp", f"pytorch_single_node_training_{rand_int}.yaml")
    pod_name = f"pytorch-single-node-training-{rand_int}"

    args = "git clone https://github.com/pytorch/examples.git && python examples/mnist/main.py"

    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    cpu_limit = 72
    cpu_limit = str(int(cpu_limit) / 2)

    search_replace_dict = {
        "<POD_NAME>": pod_name,
        "<CONTAINER_NAME>": pytorch_training,
        "<ARGS>": args,
        "<CPU_LIMIT>": cpu_limit,
    }

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.SINGLE_NODE_TRAINING_TEMPLATE_PATH, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            pytorch_out = run("kubectl logs {}".format(pod_name)).stdout
            if "Accuracy" in pytorch_out:
                training_result = True
            else:
                eks_utils.LOGGER.info("**** training output ****")
                eks_utils.LOGGER.debug(pytorch_out)
        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))


@pytest.mark.skipif(not is_pr_context(), reason="Skip this test. It is already tested under PR context")
def test_eks_pytorch_dgl_single_node_training(pytorch_training, py3_only):

    """
    Function to create a pod using kubectl and given container image, and run
    DGL training with PyTorch backend
    Args:
        :param pytorch_training: the ECR URI
    """

    training_result = False

    rand_int = random.randint(4001, 6000)

    yaml_path = os.path.join(os.sep, "tmp", f"pytorch_single_node_training_dgl_{rand_int}.yaml")
    pod_name = f"pytorch-single-node-training-dgl-{rand_int}"

    args = (
        "git clone https://github.com/dmlc/dgl.git && "
        "cd /dgl/examples/pytorch/gcn/ && DGLBACKEND=pytorch python train.py --dataset cora"
    )

    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    cpu_limit = 72
    cpu_limit = str(int(cpu_limit) / 2)

    if "gpu" in pytorch_training:
        args = args + " --gpu 0"
    else:
        args = args + " --gpu -1"

    search_replace_dict = {
        "<POD_NAME>": pod_name,
        "<CONTAINER_NAME>": pytorch_training,
        "<ARGS>": args,
        "<CPU_LIMIT>": cpu_limit,
    }

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.SINGLE_NODE_TRAINING_TEMPLATE_PATH, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            dgl_out = run("kubectl logs {}".format(pod_name)).stdout
            if "Test accuracy" in dgl_out:
                training_result = True
            else:
                eks_utils.LOGGER.info("**** training output ****")
                eks_utils.LOGGER.debug(dgl_out)

        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))


# @pytest.mark.skipif(is_pr_context(), reason=SKIP_PR_REASON)
@pytest.mark.multinode("multinode(4)")
def test_eks_pytorch_multinode_node_training(pytorch_training, example_only):
    """
       Function to create mutliple pods using kubectl and given container image, and run Pytorch training
       Args:
           :param setup_utils: environment in which EKS tools are setup
           :param pytorch_training: the ECR URI
       """
    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    random.seed(f"{pytorch_training}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_id = random.randint(1, 6000)

    namespace = f"pytorch-multi-node-training-{unique_id}"
    app_name = f"eks-pytorch-mnist-app-{unique_id}"
    job_name = f"kubeflow-pytorch-gpu-dist-job-{unique_id}"
    num_masters = "1"
    num_workers = "3"
    gpu_limit = "1"
    backend = "gloo"
    epochs = '"10"'
    local_template_file_path = os.path.join(
        "eks",
        "eks_manifest_templates",
        "pytorch",
        "training",
        "multi_node_gpu_training.yaml"
    )
    remote_yaml_path = os.path.join(os.sep, "tmp", f"pytorch_multinode_node_training_{unique_id}.yaml")
    replace_dict = {
        "<JOB_NAME>": job_name,
        "<NUM_MASTERS>": num_masters,
        "<NUM_WORKERS>": num_workers,
        "<CONTAINER_IMAGE>": pytorch_training,
        "<BACKEND>": backend,
        "<EPOCHS>": epochs,
        "<GPU_LIMIT>": gpu_limit
    }

    eks_utils.write_eks_yaml_file_from_template(local_template_file_path, remote_yaml_path, replace_dict)
    run_eks_pytorch_multi_node_training(namespace, job_name, remote_yaml_path)


def run_eks_pytorch_multi_node_training(namespace, job_name, remote_yaml_file_path):
    """Run PyTorch distributed training on EKS using PyTorch Operator
    Args:
    namespace, job_name, remote_yaml_file_path
    """

    # Namespaces will allow parallel runs on the same cluster. Create namespace if it doesnt exist.
    does_namespace_exist = run(f"kubectl get namespace | grep {namespace}",
                               warn=True)
    if not does_namespace_exist:
        run(f"kubectl create namespace {namespace}")
    
    try:
        run(f"kubectl delete -f {remote_yaml_file_path}", warn=True)
        run(f"kubectl create -f {remote_yaml_file_path} -n {namespace}")
        training_result = is_pytorch_eks_multinode_training_complete(job_name, namespace)
        if training_result:
            run_out = run(f"kubectl logs {job_name}-master-0 -n {namespace}", warn=True).stdout
            if "accuracy" in run_out:
                training_result = True
            else:
                eks_utils.LOGGER.info("**** training output ****")
                eks_utils.LOGGER.debug(run_out)
        assert training_result, f"Training for eks pytorch multinode failed"
    finally:
        eks_utils.eks_multinode_cleanup(remote_yaml_file_path, namespace)


def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)


@retry(stop_max_attempt_number=40, wait_fixed=60000, retry_on_exception=retry_if_value_error)
def is_pytorch_eks_multinode_training_complete(job_name, namespace):
    """Function to check job and pod status for multinode training.
    A separate method is required because kubectl commands for logs and status are different with namespaces.
    Args:
        job_name: str
    """
    run_out = run(f"kubectl get pytorchjobs -n {namespace} {job_name} -o json", warn=True)
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
