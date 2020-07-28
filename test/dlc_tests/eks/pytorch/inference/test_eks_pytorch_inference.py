import os
import random

import pytest

from invoke import run

import test.test_utils.eks as eks_utils
import test.test_utils as test_utils


@pytest.mark.model("densenet")
def test_eks_pytorch_densenet_inference(pytorch_inference):
    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "gpu" if "gpu" in pytorch_inference else "cpu"
    if "eia" in pytorch_inference:
        pytest.skip("Skipping EKS Test for EIA")

    model = "pytorch-densenet=https://dlc-samples.s3.amazonaws.com/pytorch/multi-model-server/densenet/densenet.mar"
    yaml_path = os.path.join(os.sep, "tmp", f"pytorch_single_node_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"densenet-service-{processor}-{rand_int}"

    search_replace_dict = {
        "<MODELS>": model,
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": pytorch_inference
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
