import pytest
from invoke import run
import test.test_utils.eks as eks_utils


def test_eks_pytorch_single_node_training(eks_setup, pytorch_training):
    """
    Function to create a pod using kubectl and given container image, and run MXNet training
    Args:
        :param setup_utils: environment in which EKS tools are setup
        :param pytorch_training: the ECR URI
    """

    training_result = False

    print("*****************")
    run("pwd")
    print("*****************")

    template_path = (
        "test/dlc_tests/eks/eks_manifest_templates/training/single_node_training.yaml"
    )
    yaml_path = "/tmp/pytorch_single_node_training.yaml"
    pod_name = "pytorch-single-node-training"

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
        template_path, yaml_path, search_replace_dict
    )

    try:
        run("kubectl create -f {}".format(yaml_path))

        if eks_utils.is_eks_training_complete(pod_name):
            pytorch_out = run("kubectl logs {}".format(pod_name)).stdout
            if "Accuracy" in pytorch_out:
                training_result = True

        assert training_result, f"Training failed"
    finally:
        run("kubectl delete pods {}".format(pod_name))
