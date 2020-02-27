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

import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet


def model_fn(model_dir):
    sym, arg_params, aux_params = onnx_mxnet.import_model(os.path.join(model_dir, 'model.onnx'))
    mod = mx.mod.Module(symbol=sym, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', [100, 1, 28, 28])])
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    return mod
