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

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
mnist_path = os.path.join(resources_path, 'mnist')
mnist_script = os.path.join(mnist_path, 'mnist.py')
throughput_path = os.path.join(resources_path, "smdataparallel")
smdataparallel_mnist_script = os.path.join(mnist_path, 'smdataparallel_mnist_script_mode.sh')
fastai_path = os.path.join(resources_path, 'fastai')
fastai_cifar_script = os.path.join(fastai_path, 'train_cifar.py')
fastai_mnist_script = os.path.join(fastai_path, 'mnist.py')
resnet18_path = os.path.join(resources_path, 'resnet18')

data_dir = os.path.join(mnist_path, 'data')
training_dir = os.path.join(data_dir, 'training')
dist_operations_path = os.path.join(resources_path, 'distributed_operations.py')
smdebug_mnist_script = os.path.join(mnist_path, 'smdebug_mnist.py')

mnist_1d_script = os.path.join(mnist_path, 'mnist_1d.py')
model_cpu_dir = os.path.join(mnist_path, 'model_cpu')
model_cpu_1d_dir = os.path.join(model_cpu_dir, '1d')
model_gpu_dir = os.path.join(mnist_path, 'model_gpu')
model_gpu_1d_dir = os.path.join(model_gpu_dir, '1d')
call_model_fn_once_script = os.path.join(resources_path, 'call_model_fn_once.py')

ROLE = 'dummy/unused-role'
DEFAULT_TIMEOUT = 40


def get_framework_from_image_uri(image_uri):
    return (
        "huggingface_tensorflow" if "huggingface-tensorflow" in image_uri
        else "huggingface_pytorch" if "huggingface-pytorch" in image_uri
        else "mxnet" if "mxnet" in image_uri
        else "pytorch" if "pytorch" in image_uri
        else "tensorflow" if "tensorflow" in image_uri
        else None
    )


def get_framework_and_version_from_tag(image_uri):
    """
    Return the framework and version from the image tag.

    :param image_uri: ECR image URI
    :return: framework name, framework version
    """
    tested_framework = get_framework_from_image_uri(image_uri)
    allowed_frameworks = ("huggingface_tensorflow", "huggingface_pytorch", "tensorflow", "mxnet", "pytorch")

    if not tested_framework:
        raise RuntimeError(
            f"Cannot find framework in image uri {image_uri} " f"from allowed frameworks {allowed_frameworks}"
        )

    tag_framework_version = re.search(r"(\d+(\.\d+){1,2})", image_uri).groups()[0]

    return tested_framework, tag_framework_version
