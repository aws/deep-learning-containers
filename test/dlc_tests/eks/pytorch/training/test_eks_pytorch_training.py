import os
import time
import pytest
import random

import test.test_utils.eks as eks_utils

from invoke import run


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
        time.sleep(10)
        run("kubectl delete pods {}".format(pod_name))


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
        time.sleep(10)
        run("kubectl delete pods {}".format(pod_name))
