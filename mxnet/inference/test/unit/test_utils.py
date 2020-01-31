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

from mock import Mock, mock_open, patch

from sagemaker_mxnet_serving_container.utils import get_default_context, read_data_shapes

MODEL_DIR = 'foo/model'


@patch('mxnet.cpu')
def test_context(mx_cpu):
    mx_context = Mock()
    mx_cpu.return_value = mx_context

    default_context = get_default_context()

    assert default_context == mx_context


def test_read_data_shapes():
    data_name = 'foo'
    data_shape = [1]
    signature = json.dumps([{'name': data_name, 'shape': data_shape}])

    with patch('six.moves.builtins.open', mock_open(read_data=signature)):
        data_names, data_shapes = read_data_shapes(MODEL_DIR)

    assert len(data_names) == 1
    assert len(data_shapes) == 1
    assert data_names[0] == data_name
    assert data_shapes[0][1] == data_shape
