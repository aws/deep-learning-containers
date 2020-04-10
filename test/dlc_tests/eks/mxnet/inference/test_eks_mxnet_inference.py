import os
import random

import test.test_utils.eks as eks_utils
import test.test_utils as test_utils

from invoke import run


def test_eks_mxnet_squeezenet_inference(mxnet_inference):
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

    if processor is "gpu":
        search_replace_dict["<NUM_GPUS>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("mxnet", processor), yaml_path, search_replace_dict
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8080")

        assert test_utils.request_mxnet_inference_squeezenet(port=port_to_forward)
    except ValueError as excp:
        eks_utils.LOGGER.error("Service is not running: %s", excp)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")


def test_eks_mxnet_gluonnlp_inference(mxnet_inference, py3_only):
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

    if processor is "gpu":
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
    except ValueError as excp:
        eks_utils.LOGGER.error("Service is not running: %s", excp)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")