import os
import random

import pytest
from time import sleep

from invoke import run

import test.test_utils.eks as eks_utils
import test.test_utils as test_utils

@pytest.mark.model("resnet")
def test_eks_pytorch_neuron_inference(pytorch_inference, neuron_only):
    server_type = test_utils.get_inference_server_type(pytorch_inference)
    if "neuron" not in pytorch_inference:
        pytest.skip("Skipping EKS Neuron Test for EIA and Non Neuron Images")

    model = "pytorch-resnet-neuron=https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuron.mar"
    server_cmd = "/usr/local/bin/entrypoint.sh -m pytorch-resnet-neuron=https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuron.mar -t /home/model-server/config.properties"
    num_replicas = "1"
    rand_int = random.randint(4001, 6000)
    processor = "neuron"

    yaml_path = os.path.join(os.sep, "tmp", f"pytorch_single_node_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"resnet-{processor}-{rand_int}"

    search_replace_dict = {
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": pytorch_inference,
        "<SERVER_TYPE>": server_type,
        "<SERVER_CMD>": server_cmd
    }

    search_replace_dict["<NUM_INF1S>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("pytorch", processor), yaml_path, search_replace_dict
    )
    device_plugin_path = eks_utils.get_device_plugin_path("pytorch", processor)

    try:
        # TODO - once eksctl gets the latest neuron device plugin this can be removed
        run("kubectl delete -f {}".format(device_plugin_path))
        sleep(60)
        run("kubectl apply -f {}".format(device_plugin_path))
        sleep(10)

        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8080")

        assert test_utils.request_pytorch_inference_densenet(port=port_to_forward)
    except ValueError as excp:
        run("kubectl cluster-info dump")
        eks_utils.LOGGER.error("Service is not running: %s", excp)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")

@pytest.mark.model("densenet")
def test_eks_pytorch_densenet_inference(pytorch_inference):
    server_type = test_utils.get_inference_server_type(pytorch_inference)
    if "eia" in pytorch_inference:
        pytest.skip("Skipping EKS Test for EIA")
    elif "neuron" in pytorch_inference:
        pytest.skip("Neuron specific test is run and so skipping this test for Neuron")
    elif server_type == "ts":
        model = "pytorch-densenet=https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar"
        server_cmd = "torchserve"
    else:
        model = "pytorch-densenet=https://dlc-samples.s3.amazonaws.com/pytorch/multi-model-server/densenet/densenet.mar"
        server_cmd = "multi-model-server"

    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "gpu" if "gpu" in pytorch_inference else "cpu"

    yaml_path = os.path.join(os.sep, "tmp", f"pytorch_single_node_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"densenet-service-{processor}-{rand_int}"

    search_replace_dict = {
        "<MODELS>": model,
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": pytorch_inference,
        "<SERVER_TYPE>": server_type,
        "<SERVER_CMD>": server_cmd
    }

    if processor == "gpu":
        search_replace_dict["<NUM_GPUS>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("pytorch", processor), yaml_path, search_replace_dict
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8080")

        assert test_utils.request_pytorch_inference_densenet(port=port_to_forward)
    except ValueError as excp:
        eks_utils.LOGGER.error("Service is not running: %s", excp)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")
