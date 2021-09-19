import os
import random

import pytest
from time import sleep

from invoke import run

import test.test_utils.eks as eks_utils
import test.test_utils as test_utils

@pytest.mark.model("resnet50")
def test_eks_mxnet_neuron_inference(mxnet_inference, neuron_only):
    if "eia" in mxnet_inference or "neuron" not in mxnet_inference:
        pytest.skip("Skipping EKS Neuron Test for EIA and Non Neuron Images")
    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "neuron"

    server_cmd = "/usr/local/bin/entrypoint.sh -m mxnet-resnet50=https://aws-dlc-sample-models.s3.amazonaws.com/mxnet/Resnet50-neuron.mar -t /home/model-server/config.properties"
    yaml_path = os.path.join(os.sep, "tmp", f"mxnet_single_node_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"resnet50-{processor}-{rand_int}"

    search_replace_dict = {
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": mxnet_inference,
        "<SERVER_CMD>": server_cmd
    }

    search_replace_dict["<NUM_INF1S>"] = "1"


    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("mxnet", processor), yaml_path, search_replace_dict
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8080")

        assert test_utils.request_mxnet_inference(port=port_to_forward, model="mxnet-resnet50")
    finally:
        run("kubectl cluster-info dump")
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")

@pytest.mark.model("squeezenet")
def test_eks_mxnet_squeezenet_inference(mxnet_inference):
    if "eia" in mxnet_inference or "neuron" in mxnet_inference:
        pytest.skip("Skipping EKS Test for EIA and neuron images")
    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "gpu" if "gpu" in mxnet_inference else "cpu"

    model = "squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model"
    yaml_path = os.path.join(os.sep, "tmp", f"mxnet_single_node_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"squeezenet-service-{rand_int}"

    search_replace_dict = {
        "<MODELS>": model,
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": mxnet_inference
    }

    if processor == "gpu":
        search_replace_dict["<NUM_GPUS>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("mxnet", processor), yaml_path, search_replace_dict
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8080")

        assert test_utils.request_mxnet_inference(port=port_to_forward)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")


@pytest.mark.skip("Flaky test. Same test passes on EC2. Fails for gpu-inference for mx1.7. Refer: https://github.com/aws/deep-learning-containers/issues/587")
@pytest.mark.integration("gluonnlp")
@pytest.mark.model("bert_sst")
def test_eks_mxnet_gluonnlp_inference(mxnet_inference, py3_only):
    if "eia" in mxnet_inference:
        pytest.skip("Skipping EKS Test for EIA")
    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "gpu" if "gpu" in mxnet_inference else "cpu"

    model = "https://aws-dlc-sample-models.s3.amazonaws.com/bert_sst/bert_sst.mar"
    yaml_path = os.path.join(os.sep, "tmp", f"mxnet_single_node_gluonnlp_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"gluonnlp-service-{processor}-{rand_int}"

    search_replace_dict = {
        "<MODELS>": model,
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": mxnet_inference
    }

    if processor == "gpu":
        search_replace_dict["<NUM_GPUS>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("mxnet", processor), yaml_path, search_replace_dict
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8080")

        assert test_utils.request_mxnet_inference_gluonnlp(port=port_to_forward)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")
