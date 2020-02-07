"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""

import os

# Function Status Codes
SUCCESS = 0
FAIL = 1
NOT_BUILT = -1

# Left and right padding between text and margins in output
PADDING = 1

# Docker connections
DOCKER_URL = "unix://var/run/docker.sock"

STATUS_MESSAGE = {SUCCESS: "Success", FAIL: "Failed", NOT_BUILT: "Not Built"}

BUILD_CONTEXT = os.environ.get("BUILD_CONTEXT", "DEV")

METRICS_NAMESPACE = "dlc-metrics-to-be-deleted"
