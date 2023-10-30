# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import re
import boto3

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
gpt2_path = os.path.join(resources_path, "gpt2")
gpt2_script = os.path.join(gpt2_path, "train_gpt_simple.py")
mnist_path = os.path.join(resources_path, "mnist")
mnist_script = os.path.join(mnist_path, "mnist.py")
throughput_path = os.path.join(resources_path, "smdataparallel")
smdataparallel_mnist_script = os.path.join(mnist_path, "smdataparallel_mnist_script_mode.sh")
fastai_path = os.path.join(resources_path, "fastai")
fastai_cifar_script = os.path.join(fastai_path, "train_cifar.py")
fastai_mnist_script = os.path.join(fastai_path, "mnist.py")
resnet18_path = os.path.join(resources_path, "resnet18")

data_dir = os.path.join(mnist_path, "data")
training_dir = os.path.join(data_dir, "training")
dist_operations_path = os.path.join(resources_path, "distributed_operations.py")
neuron_allreduce_path = os.path.join(resources_path, "neuron", "all_reduce")
neuron_mlp_path = os.path.join(resources_path, "neuron", "mlp")
smdebug_mnist_script = os.path.join(mnist_path, "smdebug_mnist.py")
smppy_mnist_script = os.path.join(mnist_path, "smppy_mnist.py")

mnist_1d_script = os.path.join(mnist_path, "mnist_1d.py")
model_cpu_dir = os.path.join(mnist_path, "model_cpu")
model_cpu_1d_dir = os.path.join(model_cpu_dir, "1d")
model_gpu_dir = os.path.join(mnist_path, "model_gpu")
model_gpu_1d_dir = os.path.join(model_gpu_dir, "1d")
call_model_fn_once_script = os.path.join(resources_path, "call_model_fn_once.py")

ROLE = "dummy/unused-role"
DEFAULT_TIMEOUT = 40


def get_framework_from_image_uri(image_uri):
    return (
        "huggingface_tensorflow_trcomp"
        if "huggingface-tensorflow-trcomp" in image_uri
        else "huggingface_tensorflow"
        if "huggingface-tensorflow" in image_uri
        else "huggingface_pytorch_trcomp"
        if "huggingface-pytorch-trcomp" in image_uri
        else "huggingface_pytorch"
        if "huggingface-pytorch" in image_uri
        else "mxnet"
        if "mxnet" in image_uri
        else "pytorch"
        if "pytorch" in image_uri
        else "tensorflow"
        if "tensorflow" in image_uri
        else None
    )


def get_framework_and_version_from_tag(image_uri):
    """
    Return the framework and version from the image tag.

    :param image_uri: ECR image URI
    :return: framework name, framework version
    """
    tested_framework = get_framework_from_image_uri(image_uri)
    allowed_frameworks = (
        "huggingface_tensorflow_trcomp",
        "huggingface_pytorch_trcomp",
        "huggingface_tensorflow",
        "huggingface_pytorch",
        "tensorflow",
        "mxnet",
        "pytorch",
    )

    if not tested_framework:
        raise RuntimeError(
            f"Cannot find framework in image uri {image_uri} "
            f"from allowed frameworks {allowed_frameworks}"
        )

    tag_framework_version = re.search(r"(\d+(\.\d+){1,2})", image_uri).groups()[0]

    return tested_framework, tag_framework_version


def get_region_from_image_uri(image_uri):
    """
    Find the region where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Region Name
    """
    region_pattern = r"(us(-gov)?|af|ap|ca|cn|eu|il|me|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()


def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]


def get_repository_and_tag_from_image_uri(image_uri):
    """
    Return the name of the repository holding the image

    :param image_uri: URI of the image
    :return: <str> repository name
    """
    repository_uri, tag = image_uri.split(":")
    _, repository_name = repository_uri.split("/")
    return repository_name, tag


def get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri):
    """
    Uses ecr describe to generate all the tags of an image.

    :param ecr_client: boto3 Client for ECR
    :param image_uri: str Image URI
    :return: list, All the image tags
    """
    account_id = get_account_id_from_image_uri(image_uri)
    image_repo_name, image_tag = get_repository_and_tag_from_image_uri(image_uri)
    response = ecr_client.describe_images(
        registryId=account_id,
        repositoryName=image_repo_name,
        imageIds=[
            {"imageTag": image_tag},
        ],
    )
    return response["imageDetails"][0]["imageTags"]


def get_cuda_version_from_tag(image_uri):
    """
    Return the cuda version from the image tag as cuXXX
    :param image_uri: ECR image URI
    :return: cuda version as cuXXX
    """
    cuda_framework_version = None
    cuda_str = ["cu", "gpu"]
    image_region = get_region_from_image_uri(image_uri)
    ecr_client = boto3.Session(region_name=image_region).client("ecr")
    all_image_tags = get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri)

    for image_tag in all_image_tags:
        if all(keyword in image_tag for keyword in cuda_str):
            cuda_framework_version = re.search(r"(cu\d+)-", image_tag).groups()[0]
            return cuda_framework_version

    if "gpu" in image_uri:
        raise CudaVersionTagNotFoundException()
    else:
        return None
