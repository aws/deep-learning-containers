# Copyright 2018-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import subprocess
import sys
import torch
from packaging.version import Version
TORCH_VERSION = torch.__version__
pre_ptbackend = Version(TORCH_VERSION) < Version("1.10")

if pre_ptbackend:
    script = 'smdataparallel_throughput_pre_ptbackend.py'
else:
    script = 'smdataparallel_throughput_post_ptbackend.py'
exe = 'python'

cmd = [exe] + [script] + sys.argv[1:]
cmd = ' '.join(cmd)

subprocess.run(cmd, shell=True)