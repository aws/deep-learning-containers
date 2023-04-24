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


def model_fn(model_dir):
    lock_file = os.path.join(model_dir, 'model_fn.lock.{}'.format(os.getpid()))
    if os.path.exists(lock_file):
        raise RuntimeError('model_fn called more than once (lock: {})'.format(lock_file))

    open(lock_file, 'a').close()

    return 'model'


def input_fn(data, content_type):
    return data


def predict_fn(data, model):
    return b'output'


def output_fn(prediction, accept):
    return prediction
