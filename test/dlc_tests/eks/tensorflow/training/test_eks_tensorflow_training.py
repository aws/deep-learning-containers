import os
import random

import pytest

from invoke import run

import test.test_utils.eks as eks_utils


@pytest.mark.model("darknet53")
def test_eks_tensorflow_single_node_training(tensorflow_training):
    """
    Function to create a pod using kubectl and given container image, and run MXNet training

    :param setup_utils: environment in which EKS tools are setup
    :param tensorflow_training: the ECR URI
    """

    training_result = False

    rand_int = random.randint(4001, 6000)

    yaml_path = os.path.join(os.sep, "tmp", f"tensorflow_single_node_training_{rand_int}.yaml")
    pod_name = f"tensorflow-single-node-training-{rand_int}"

    args = ("aws s3 cp s3://mzanur-data/tf_records /imagenet_data/ --recursive "
            "&& pip install tensorflow_addons && pip install tqdm "
            "&& git clone https://github.com/anuragrs/deep-learning-models.git "
            "&& cd deep-learning-models/models/vision/classification/"
            "&& mpirun -np 8 -H localhost:8 -map-by slot -x NCCL_DEBUG=INFO -x TF_XLA_FLAGS=--tf_xla_cpu_global_jit -mca btl ^vader -mca btl_tcp_if_exclude tun0,docker0,lo --bind-to none --allow-run-as-root python train_backbone.py --train_data_dir /imagenet_data/train/ --validation_data_dir /imagenet_data/validation -b 128 --num_epochs 5 --model darknet53 --schedule cosine")

    # TODO: Change hardcoded value to read a mapping from the EKS cluster instance.
    cpu_limit = 72
    cpu_limit = str(int(cpu_limit) / 2)

    search_replace_dict = {
        "<POD_NAME>": pod_name,
        "<CONTAINER_NAME>": tensorflow_training,
        "<ARGS>": args,
        "<CPU_LIMIT>": cpu_limit,
    }

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.SINGLE_NODE_TRAINING_TEMPLATE_PATH, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            tensorflow_out = run("kubectl logs {}".format(pod_name)).stdout
            if "Test accuracy" in tensorflow_out:
                training_result = True
            else:
                eks_utils.LOGGER.info("**** training output ****")
                eks_utils.LOGGER.debug(tensorflow_out)

        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))
