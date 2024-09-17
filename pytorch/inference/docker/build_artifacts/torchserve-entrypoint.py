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
import signal


def reap_zombies(signum, frame):
    """
    Signal handler to reap zombie processes.
    """
    while True:
        try:
            # Wait for any child process to terminate
            pid, status = os.waitpid(-1, os.WNOHANG)
            # If no more children, exit the loop
            if pid == 0:
                break
        except ChildProcessError:
            # No child processes left
            break


# Set up the signal handler for SIGCHLD
signal.signal(signal.SIGCHLD, reap_zombies)

if sys.argv[1] == "serve":
    from sagemaker_pytorch_serving_container import serving

    serving.main()
else:
    subprocess.check_call(shlex.split(" ".join(sys.argv[1:])))

# prevent docker exit
subprocess.call(["tail", "-f", "/dev/null"])
