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

import os.path
import subprocess
import shlex
import sys

pip_list_output = subprocess.Popen(["pip", "list"], stdout=subprocess.PIPE)
tf_serving_output = subprocess.check_output(
    ("grep", "tensorflow"), stdin=pip_list_output.stdout
).decode("utf-8")
if "gpu" in tf_serving_output:
    # run compat mounting by default
    try:
        subprocess.run(["bash", "-m", "/usr/local/bin/start_cuda_compat.sh"])
    except Exception as e:
        print(f"Error running script: {e}")

if not os.path.exists("/opt/ml/input/config"):
    subprocess.call(["python", "/usr/local/bin/deep_learning_container.py", "&>/dev/null", "&"])

subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
