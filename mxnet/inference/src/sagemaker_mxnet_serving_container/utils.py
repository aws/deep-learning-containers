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

import json

import mxnet as mx


def get_default_context():
    """Get the default context.

    Returns:
        context : The corresponding CPU context.

    """
    # TODO mxnet ctx - better default, allow user control
    return mx.cpu()


def read_data_shapes(path, preferred_batch_size=1):
    """Read the data name and data shape required by the MXNet module.

    Args:
        path (str): an MXNet NDArray that is the result of a prediction
        preferred_batch_size (int): the accept content type expected by the client

    Returns:
        tuple: A list of names for data required by the module along with
            a list of (name, shape) pairs specifying the data inputs to this module.

    """
    with open(path, 'r') as f:
        signatures = json.load(f)

    data_names = []
    data_shapes = []

    for s in signatures:
        name = s['name']
        data_names.append(name)

        shape = s['shape']

        if preferred_batch_size:
            shape[0] = preferred_batch_size

        data_shapes.append((name, shape))

    return data_names, data_shapes
