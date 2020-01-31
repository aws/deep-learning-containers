# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import os

from mock import Mock, mock_open, patch

from sagemaker_mxnet_container import training_utils

SCHEDULER_HOST = 'host-1'
WORKER_HOST = 'host-2'
MODEL_DIR = 'foo/model'


@patch('json.dump')
@patch('os.environ', {'SM_CURRENT_HOST': SCHEDULER_HOST, 'SM_HOSTS': json.dumps([SCHEDULER_HOST])})
def test_save_single_machine(json_dump):
    model = Mock()
    model.data_shapes = []

    with patch('six.moves.builtins.open', mock_open()):
        training_utils.save(MODEL_DIR, model)

    model.symbol.save.assert_called_with(os.path.join(MODEL_DIR, 'model-symbol.json'))
    model.save_params.assert_called_with(os.path.join(MODEL_DIR, 'model-0000.params'))
    json_dump.assert_called_once


@patch('json.dump')
def test_save_distributed(json_dump):
    model = Mock()
    model.data_shapes = []

    with patch('six.moves.builtins.open', mock_open()):
        training_utils.save(MODEL_DIR, model, current_host=SCHEDULER_HOST,
                            hosts=[SCHEDULER_HOST, WORKER_HOST])

    model.symbol.save.assert_called_with(os.path.join(MODEL_DIR, 'model-symbol.json'))
    model.save_params.assert_called_with(os.path.join(MODEL_DIR, 'model-0000.params'))
    json_dump.assert_called_once


def test_save_for_non_scheduler_host():
    model = Mock()
    training_utils.save(MODEL_DIR, model, current_host=WORKER_HOST,
                        hosts=[SCHEDULER_HOST, WORKER_HOST])

    model.symbol.save.assert_not_called
    model.save_params.assert_not_called


def test_single_machine_scheduler_host():
    assert training_utils.scheduler_host([SCHEDULER_HOST]) == SCHEDULER_HOST


def test_distributed_scheduler_host():
    assert training_utils.scheduler_host([SCHEDULER_HOST, WORKER_HOST]) == SCHEDULER_HOST
