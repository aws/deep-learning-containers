import os
import random

import test.test_utils.eks as eks_utils
import test.test_utils as test_utils

from invoke import run


def test_eks_tensorflow_half_plus_two_inference(tensorflow_inference):
    num_replicas = "1"

    rand_int = random.randint(4001, 6000)

    processor = "gpu" if "gpu" in tensorflow_inference else "cpu"

    model_name = f"saved_model_half_plus_two_{processor}"
    yaml_path = os.path.join(os.sep, "tmp", f"tensorflow_single_node_{processor}_inference_{rand_int}.yaml")
    inference_service_name = selector_name = f"half-plus-two-service-{processor}-{rand_int}"

    search_replace_dict = {
        "<MODEL_NAME>": model_name,
        "<MODEL_BASE_PATH>": f"s3://tensoflow-trained-models/{model_name}",
        "<NUM_REPLICAS>": num_replicas,
        "<SELECTOR_NAME>": selector_name,
        "<INFERENCE_SERVICE_NAME>": inference_service_name,
        "<DOCKER_IMAGE_BUILD_ID>": tensorflow_inference
    }

    if processor is "gpu":
        search_replace_dict["<NUM_GPUS>"] = "1"

    eks_utils.write_eks_yaml_file_from_template(
        eks_utils.get_single_node_inference_template_path("tensorflow", processor), yaml_path, search_replace_dict
    )

    try:
        run("kubectl apply -f {}".format(yaml_path))

        port_to_forward = random.randint(49152, 65535)

        if eks_utils.is_service_running(selector_name):
            eks_utils.eks_forward_port_between_host_and_container(selector_name, port_to_forward, "8500")

        assert test_utils.request_tensorflow_inference(model_name=model_name, port=port_to_forward)
    except ValueError as excp:
        eks_utils.LOGGER.error("Service is not running: %s", excp)
    finally:
        run(f"kubectl delete deployment {selector_name}")
        run(f"kubectl delete service {selector_name}")
