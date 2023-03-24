# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

resources_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources'))
mnist_path = os.path.join(resources_path, 'mnist')
resnet_path = os.path.join(resources_path, 'resnet')
mnist_script = os.path.join(mnist_path, 'mnist.py')
data_dir = os.path.join(mnist_path, 'data')
training_dir = os.path.join(data_dir, 'training')
cpu_sub_dir = 'model_cpu'
gpu_sub_dir = 'model_gpu'
eia_sub_dir = 'model_eia'
neuron_sub_dir = 'model_neuron'
code_sub_dir = 'code'

model_cpu_dir = os.path.join(mnist_path, cpu_sub_dir)
mnist_cpu_script = os.path.join(model_cpu_dir, code_sub_dir ,'mnist.py')
model_cpu_1d_dir = os.path.join(model_cpu_dir, '1d')
mnist_1d_script = os.path.join(model_cpu_1d_dir, code_sub_dir, 'mnist_1d.py')
model_gpu_dir = os.path.join(mnist_path, gpu_sub_dir)
mnist_gpu_script = os.path.join(model_gpu_dir, code_sub_dir, 'mnist.py')
model_gpu_1d_dir = os.path.join(model_gpu_dir, '1d')
model_eia_dir = os.path.join(mnist_path, eia_sub_dir)
mnist_eia_script = os.path.join(model_eia_dir, 'mnist.py')
model_neuron_dir = os.path.join(resnet_path, neuron_sub_dir)
resnet_neuron_script = os.path.join(model_neuron_dir, code_sub_dir, 'resnet18.py')
resnet_neuron_input = os.path.join(model_neuron_dir, 'cat.jpg')
resnet_neuron_image_list = os.path.join(model_neuron_dir, 'imagenet1000_clsidx_to_labels.txt')
call_model_fn_once_script = os.path.join(resources_path, code_sub_dir, 'call_model_fn_once.py')

ROLE = "dummy/unused-role"
DEFAULT_TIMEOUT = 20

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
