# Copyright 2019-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from pathlib import Path
import shlex
import subprocess
import sys

from sagemaker_inference import environment

SAI_MODEL_CACHE_FILE = os.path.join(
    environment.model_dir, os.getenv("SAI_MODEL_CACHE_FILE", "stabilityai-model-cache.tar")
)
SAI_MODEL_CACHE_PATH = os.getenv("SAI_MODEL_CACHE_PATH", "/tmp/cache")
SAI_MODEL_CACHE_STATUS_FILE = os.path.join(SAI_MODEL_CACHE_PATH, ".model-cache-unpacked")
if os.path.exists(SAI_MODEL_CACHE_FILE) and not os.path.exists(SAI_MODEL_CACHE_STATUS_FILE):
    os.makedirs(SAI_MODEL_CACHE_PATH, exist_ok=True)
    subprocess.check_call(
        [
            "tar",
            "x",
            "-z" if SAI_MODEL_CACHE_FILE.endswith(".gz") else "",
            "-f",
            SAI_MODEL_CACHE_FILE,
            "-C",
            SAI_MODEL_CACHE_PATH,
        ]
    )
    Path(SAI_MODEL_CACHE_STATUS_FILE).touch()

if sys.argv[1] == "serve":
    from sagemaker_pytorch_serving_container import serving

    serving.main()
else:
    subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

# prevent docker exit
subprocess.call(["tail", "-f", "/dev/null"])
