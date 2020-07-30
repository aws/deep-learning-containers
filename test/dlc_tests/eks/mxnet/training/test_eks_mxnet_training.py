import os
import random
import re

import pytest

from invoke import run
from invoke.context import Context

import test.test_utils.eks as eks_utils


@pytest.mark.model("mnist")
def test_eks_mxnet_single_node_training(mxnet_training):
    """
    Function to create a pod using kubectl and given container image, and run MXNet training
    Args:
        :param mxnet_training: the ECR URI
    """

    training_result = False

    rand_int = random.randint(4001, 6000)

    framework_version_search = re.search(r"\d+(\.\d+){2}", mxnet_training)
    framework_version = framework_version_search.group()
    if not framework_version_search:
        framework_version_search = re.search(r"\d+\.\d+", mxnet_training)
        framework_version = framework_version_search.group() + ".0"

    yaml_path = os.path.join(os.sep, "tmp", f"mxnet_single_node_training_{rand_int}.yaml")
    pod_name = f"mxnet-single-node-training-{rand_int}"

    args = (
        f"git clone -b {framework_version} https://github.com/apache/incubator-mxnet.git && python "
        f"/incubator-mxnet/example/image-classification/train_mnist.py"
    )

    processor_type = "gpu" if "gpu" in mxnet_training else "cpu"
    args = args + " --gpus 0" if processor_type == "gpu" else args

    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    cpu_limit = 72
    cpu_limit = str(int(cpu_limit) / 2)

    search_replace_dict = {
        "<POD_NAME>": pod_name,
        "<CONTAINER_NAME>": mxnet_training,
        "<ARGS>": args,
        "<CPU_LIMIT>": cpu_limit,
    }

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.SINGLE_NODE_TRAINING_TEMPLATE_PATH, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            mxnet_out = run("kubectl logs {}".format(pod_name)).stdout
            if "Epoch[19] Validation-accuracy" in mxnet_out:
                training_result = True
            else:
                eks_utils.LOGGER.info("**** training output ****")
                eks_utils.LOGGER.debug(mxnet_out)

        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))


@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
def test_eks_mxnet_dgl_single_node_training(mxnet_training, py3_only):

    """
    Function to create a pod using kubectl and given container image, and run
    DGL training with MXNet backend
    Args:
        :param mxnet_training: the ECR URI
    """

    training_result = False
    ctx = Context()
    rand_int = random.randint(4001, 6000)

    yaml_path = os.path.join(os.sep, "tmp", f"mxnet_single_node_training_dgl_{rand_int}.yaml")
    pod_name = f"mxnet-single-node-training-dgl-{rand_int}"

    dgl_branch = eks_utils.get_dgl_branch(ctx, mxnet_training)

    args = (
        f"git clone -b {dgl_branch} https://github.com/dmlc/dgl.git && "
        f"cd /dgl/examples/mxnet/gcn/ && DGLBACKEND=mxnet python train.py --dataset cora"
    )

    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    cpu_limit = 72
    cpu_limit = str(int(cpu_limit) / 2)

    if "gpu" in mxnet_training:
        args = args + " --gpu 0"
    else:
        args = args + " --gpu -1"

    search_replace_dict = {
        "<POD_NAME>": pod_name,
        "<CONTAINER_NAME>": mxnet_training,
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


@pytest.mark.integration("gluonnlp")
@pytest.mark.model("TextCNN")
def test_eks_mxnet_gluonnlp_single_node_training(mxnet_training, py3_only):

    """
    Function to create a pod using kubectl and given container image, and run
    DGL training with MXNet backend
    Args:
        :param mxnet_training: the ECR URI
    """

    training_result = False

    rand_int = random.randint(4001, 6000)

    yaml_path = os.path.join(os.sep, "tmp", f"mxnet_single_node_training_gluonnlp_{rand_int}.yaml")
    pod_name = f"mxnet-single-node-training-gluonnlp-{rand_int}"

    args = (
        "git clone -b master https://github.com/dmlc/gluon-nlp.git && "
        "cd gluon-nlp && git checkout v0.9.0 &&"
        "cd ./scripts/sentiment_analysis/ &&"
        "python sentiment_analysis_cnn.py --batch_size 50 --epochs 20 --dropout 0.5 "
        "--model_mode multichannel --data_name TREC"
    )

    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    cpu_limit = 72
    cpu_limit = str(int(cpu_limit) / 2)

    if "gpu" in mxnet_training:
        args = args + " --gpu 0"

    search_replace_dict = {
        "<POD_NAME>": pod_name,
        "<CONTAINER_NAME>": mxnet_training,
        "<ARGS>": args,
        "<CPU_LIMIT>": cpu_limit,
    }

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.SINGLE_NODE_TRAINING_TEMPLATE_PATH, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            gluonnlp_out = run("kubectl logs {}".format(pod_name)).stdout

            results = re.search(r"test acc ((?:\d*\.\d+)|\d+)", gluonnlp_out)
            if results is not None:
                accuracy = float(results.groups()[0])

                if accuracy >= 0.75:
                    eks_utils.LOGGER.info(
                        "GluonNLP EKS test succeeded with accuracy {} >= 0.75".format(
                            accuracy
                        )
                    )
                    training_result = True
                else:
                    eks_utils.LOGGER.info(
                        "GluonNLP EKS test FAILED with accuracy {} < 0.75".format(
                            accuracy
                        )
                    )
                    eks_utils.LOGGER.debug(gluonnlp_out)

        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))
