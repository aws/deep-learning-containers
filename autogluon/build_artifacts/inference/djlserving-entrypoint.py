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

import shlex
import subprocess
import sys
import os


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        subprocess.check_call(["/usr/local/bin/setup_model.sh"])
        os.execv("/usr/bin/djl-serving", ["djl-serving", "-f", "/home/model-server/config.properties"])
    elif len(sys.argv) > 1:
        subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))
    else:
        subprocess.check_call(["/usr/local/bin/setup_model.sh"])
        os.execv("/usr/bin/djl-serving", ["djl-serving", "-f", "/home/model-server/config.properties"])