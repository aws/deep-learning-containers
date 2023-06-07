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

RESOURCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))

# these regions have some p2 instances, but not enough for automated testing
NO_P2_REGIONS = [
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "ap-northeast-3",
    "il-central-1",
]
NO_P3_REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]
NO_P4_REGIONS = [
    "ap-northeast-3",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-south-1",
    "ca-central-1",
    "eu-central-1",
    "eu-west-2",
    "us-west-1",
    "eu-west-3",
    "eu-north-1",
    "sa-east-1",
    "ap-east-1",
    "me-south-1",
    "cn-northwest-1",
    "eu-south-1",
    "af-south-1",
    "il-central-1",
]

MODEL_SUCCESS_FILES = {
    "output": ["success"],
    "model": ["model-symbol.json", "model-shapes.json", "model-0000.params"],
}
