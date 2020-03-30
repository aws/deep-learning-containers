import pytest
from invoke import run
import test.test_utils.eks as eks_utils

#@pytest.mark.skip(reason="Ignoring for now")
def test_eks_mxnet_single_node_training(eks_setup, mxnet_training):
    """
    Function to create a pod using kubectl and given container image, and run MXNet training
    Args:
        :param setup_utils: environment in which EKS tools are setup
        :param mxnet_training: the ECR URI
    """

    training_result = False

    print("*****************")
    run("pwd")
    print("*****************")

    template_path = (
        "test/dlc_tests/eks/eks_manifest_templates/training/single_node_training.yaml"
    )
    yaml_path = "/tmp/mxnet_single_node_training.yaml"
    pod_name = "mxnet-single-node-training"

    args = (
        "git clone https://github.com/apache/incubator-mxnet.git && python "
        "/incubator-mxnet/example/image-classification/train_mnist.py"
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
        template_path, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            mxnet_out = run("kubectl logs {}".format(pod_name)).stdout
            if "Epoch[19] Validation-accuracy" in mxnet_out:
                training_result = True

        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))
