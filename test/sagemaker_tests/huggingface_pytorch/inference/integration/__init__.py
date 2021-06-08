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


model_dir = os.path.join(resources_path, 'tiny-distilbert-sst-2')
# TODO: current local test, tests without custom script
# mnist_cpu_script = oos.path.join(resources_path, 'tiny-distilbert-sst-2','inference.py')


ROLE = "dummy/unused-role"
DEFAULT_TIMEOUT = 20

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))
