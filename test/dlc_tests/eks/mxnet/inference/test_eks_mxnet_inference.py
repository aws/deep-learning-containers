import pytest
import random
from invoke import run
import test.test_utils.eks as eks_utils

# def run_eks_mxnet_squeezenet_inference(mxnet_inference, processor, eks_gpus_per_worker):
#
#     num_replicas = "1"
#
#     processor = "gpu" if "gpu" in mxnet_inference else "cpu"
#
#     namespace = "{}-single-node-inference".format("mxnet")
#     selector_name = "squeezenet-service"
#
#     mxnet_inference_search_replace_dict = {
#         "<MODELS>": "squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model",
#         "<NUM_REPLICAS>": num_replicas,
#         "<SELECTOR_NAME>": selector_name
#         }
#
#     eks_utils.run_inference_service_on_eks("mxnet", processor, namespace, selector_name, eks_gpus_per_worker,
#                                         mxnet_inference_search_replace_dict)
#
#     port_to_forward = random.randint(8500, 8599)
#
#     eks_utils.eks_forward_port_between_host_and_container(namespace, selector_name, port_to_forward, "8080")
#
#     # Run inference
#     return helper.test_mxnet_inference_squeezenet(port=port_to_forward)