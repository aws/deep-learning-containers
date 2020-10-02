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

from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import onnx
from onnx import checker


def _read_data_shapes(path, preferred_batch_size=1):
    with open(path, 'r') as f:
        signature = json.load(f)

    data_shapes = []

    for s in signature:
        shape = s['shape']

        if preferred_batch_size:
            shape[0] = preferred_batch_size

        data_shapes.append(shape)

    return data_shapes


def _assert_onnx_validity(model_path):
    model_proto = onnx.load_model(model_path)
    checker.check_graph(model_proto.graph)


def main(training_dir, model_dir):
    sym = os.path.join(training_dir, 'model-symbol.json')
    params = os.path.join(training_dir, 'model-0000.params')
    data_shapes = _read_data_shapes(os.path.join(training_dir, 'model-shapes.json'))

    output_path = os.path.join(model_dir, 'model.onnx')
    converted_path = onnx_mxnet.export_model(sym, params, data_shapes, np.float32, output_path)
    _assert_onnx_validity(converted_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.train, args.model_dir)
