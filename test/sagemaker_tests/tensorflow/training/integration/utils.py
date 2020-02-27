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
import random
import time

import pytest


def unique_name_from_base(base, max_length=63):
    unique = '%04x' % random.randrange(16**4)  # 4-digit hex
    ts = str(int(time.time()))
    available_length = max_length - 2 - len(ts) - len(unique)
    trimmed = base[:available_length]
    return '{}-{}-{}'.format(trimmed, ts, unique)


@pytest.fixture(params=os.environ['TEST_PY_VERSIONS'].split(','))
def py_version(request):
    return request.param


@pytest.fixture(params=os.environ['TEST_PROCESSORS'].split(','))
def processor(request):
    return request.param
