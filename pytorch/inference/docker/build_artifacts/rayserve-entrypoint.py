# Copyright 2019-2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import shlex
import subprocess
import sys

# Auto-start Ray cluster and Ray Serve
subprocess.run(
    [
        "ray",
        "start",
        "--head",
        "--disable-usage-stats",
        "--dashboard-host",
        "0.0.0.0",
        "--dashboard-port",
        "8265",
    ],
    check=True,
)

subprocess.run(["serve", "start", "--http-host", "0.0.0.0", "--http-port", "8000"], check=True)

if len(sys.argv) > 1:
    subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
else:
    subprocess.call(["tail", "-f", "/dev/null"])
