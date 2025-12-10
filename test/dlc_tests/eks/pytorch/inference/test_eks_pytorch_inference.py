import os
import random

import pytest
from time import sleep
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from invoke import run

import test.test_utils.eks as eks_utils
import test.test_utils as test_utils


def __run_pytorch_neuron_inference(image, model_name, model_url, processor):
    server_type = test_utils.get_inference_server_type(image)

    model = f"{model_name}={model_url}"
    server_cmd = "torchserve"

    num_replicas = "1"
    rand_int = random.randint(4001, 6000)

    yaml_path = os.path.join(
        os.sep, "tmp", f"pytorch_single_node_{processor}_inference_{rand_int}.yaml"
    )
    inference_service_name = selector_name = f"resnet-{processor}-{rand_int}"

    search_replace_dict = {
        "<MODELS>": model,
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": image,
        "<SERVER_TYPE>": server_type,
        "<SERVER_CMD>": server_cmd,
    }

    search_replace_dict["<NUM_INF1S>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("pytorch", processor),
        yaml_path,
        search_replace_dict,
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(
                selector_name, port_to_forward, "8080"
            )

        assert test_utils.request_pytorch_inference_densenet(
            port=port_to_forward, server_type=server_type, model_name=model_name
        )
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")


@pytest.mark.model("resnet")
@pytest.mark.team("neuron")
def test_eks_pytorch_neuron_inference(pytorch_inference_neuron):
    __run_pytorch_neuron_inference(
        pytorch_inference_neuron,
        "pytorch-resnet-neuron",
        "https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuron.mar",
        "neuron",
    )


@pytest.mark.skip("No trn1 in the EKS cluster, disabled temporarily")
@pytest.mark.model("resnet")
@pytest.mark.team("neuron")
def test_eks_pytorch_neuronx_inference(pytorch_inference_neuronx):
    __run_pytorch_neuron_inference(
        pytorch_inference_neuronx,
        "pytorch-resnet-neuronx",
        "https://aws-dlc-sample-models.s3.amazonaws.com/pytorch/Resnet50-neuronx.mar",
        "neuronx",
    )


@pytest.mark.model("densenet")
@pytest.mark.team("conda")
def test_eks_pytorch_densenet_inference(pytorch_inference):
    _, version = test_utils.get_framework_and_version_from_tag(pytorch_inference)
    disable_token_auth = False
    # PT 2.4 and above require the disable token auth flag
    # Using workaround from https://github.com/facebookresearch/AnimatedDrawings/issues/295
    if Version(version) in SpecifierSet(">=2.4"):
        disable_token_auth = True
    __test_eks_pytorch_densenet_inference(pytorch_inference, disable_token_auth=disable_token_auth)


@pytest.mark.model("densenet")
def test_eks_pytorch_densenet_inference_graviton(pytorch_inference_graviton, cpu_only):
    _, version = test_utils.get_framework_and_version_from_tag(pytorch_inference_graviton)
    disable_token_auth = False
    # PT 2.4 and above require the disable token auth flag
    # Using workaround from https://github.com/facebookresearch/AnimatedDrawings/issues/295
    if Version(version) in SpecifierSet(">=2.4"):
        disable_token_auth = True
    __test_eks_pytorch_densenet_inference(
        pytorch_inference_graviton, disable_token_auth=disable_token_auth
    )


@pytest.mark.model("densenet")
def test_eks_pytorch_densenet_inference_arm64(pytorch_inference_arm64, cpu_only):
    _, version = test_utils.get_framework_and_version_from_tag(pytorch_inference_arm64)
    disable_token_auth = False
    # PT 2.4 and above require the disable token auth flag
    # Using workaround from https://github.com/facebookresearch/AnimatedDrawings/issues/295
    if Version(version) in SpecifierSet(">=2.4"):
        disable_token_auth = True
    __test_eks_pytorch_densenet_inference(
        pytorch_inference_arm64, disable_token_auth=disable_token_auth
    )


def __test_eks_pytorch_densenet_inference(pytorch_inference, disable_token_auth=False):
    server_type = test_utils.get_inference_server_type(pytorch_inference)
    if server_type == "ts":
        model = "pytorch-densenet=https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar"
        server_cmd = "torchserve"
    else:
        model = "pytorch-densenet=https://dlc-samples.s3.amazonaws.com/pytorch/multi-model-server/densenet/densenet.mar"
        server_cmd = "multi-model-server"

    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "gpu" if "gpu" in pytorch_inference else "cpu"
    test_type = test_utils.get_eks_k8s_test_type_label(pytorch_inference)
    disable_token_auth = " --disable-token-auth" if disable_token_auth else ""

    yaml_path = os.path.join(
        os.sep, "tmp", f"pytorch_single_node_{processor}_inference_{rand_int}.yaml"
    )
    inference_service_name = selector_name = f"densenet-service-{processor}-{rand_int}"

    search_replace_dict = {
        "<MODELS>": model,
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": pytorch_inference,
        "<SERVER_TYPE>": server_type,
        "<SERVER_CMD>": server_cmd,
        "<DISABLE_TOKEN_AUTH>": disable_token_auth,
        "<TEST_TYPE>": test_type,
    }

    if processor == "gpu":
        search_replace_dict["<NUM_GPUS>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("pytorch", processor),
        yaml_path,
        search_replace_dict,
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(
                selector_name, port_to_forward, "8080"
            )

        assert test_utils.request_pytorch_inference_densenet(
            port=port_to_forward, server_type=server_type
        )
    except Exception as e:
        # Capture diagnostic information on failure
        print(f"\n{'='*80}")
        print(f"TEST FAILED - Capturing pod logs for {selector_name}")
        print(f"{'='*80}\n")
        
        # Get pod name
        pod_result = run(f"kubectl get pods -n default --selector=app={selector_name} -o jsonpath='{{.items[0].metadata.name}}'", warn=True, hide=True)
        pod_name = pod_result.stdout.strip()
        
        if pod_name:
            print(f"Pod name: {pod_name}\n")
            
            # Get pod logs
            print("=== CONTAINER LOGS (last 100 lines) ===")
            run(f"kubectl logs {pod_name} -n default --tail=100", warn=True)
            
            # Get pod status and events
            print("\n=== POD STATUS ===")
            run(f"kubectl describe pod {pod_name} -n default | tail -50", warn=True)
        else:
            print("Could not find pod. Showing all pods:")
            run("kubectl get pods -n default", warn=True)
        
        print(f"\n{'='*80}\n")
        raise  # Re-raise the original exception
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")
