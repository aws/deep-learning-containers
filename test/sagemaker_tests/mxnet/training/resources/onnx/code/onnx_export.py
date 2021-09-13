# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import json
import os

# test the new mxnet-onnx module that released since mxnet 1.9.0
import mxnet
from mxnet import onnx as onnx_mxnet
import numpy as np
import gluoncv
import onnx
from onnx import checker

def _assert_onnx_validity(model_path):
    model_proto = onnx.load_model(model_path)
    checker.check_graph(model_proto.graph)


def main():
    prefix = './resnet18_v2'
    # input shape and type
    in_shape = (1, 3, 224, 224)
    in_dtype = 'float32'
    # download mxnet model
    gluon_model = gluoncv.model_zoo.get_model('resnet18_v2', pretrained=True)
    gluon_model.hybridize()
    # forward with dummy input and save model
    dummy_input = mxnet.nd.zeros(in_shape, dtype=in_dtype)
    gluon_model.forward(dummy_input)
    gluon_model.export(prefix, 0)

    mx_sym = prefix + '-symbol.json'
    mx_params = prefix + '-0000.params'
    onnx_file = 'model.onnx'
    in_shapes = [in_shape]
    in_dtypes = [in_dtype]
    onnx_mxnet.export_model(mx_sym, mx_params, [in_shape], [in_dtype], onnx_file)
    assert os.path.isfile('model.onnx')
    _assert_onnx_validity(onnx_file)


if __name__ == '__main__':
    main()
