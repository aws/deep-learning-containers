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

import shlex
import subprocess
import sys
import os.path

neuron_cmd = "/usr/local/bin/neuron-monitor.sh"
subprocess.check_call(shlex.split(neuron_cmd))

if sys.argv[1] == 'serve':
    user_ncgs = os.environ.get('NEURONCORE_GROUP_SIZES')
    if user_ncgs is None:
        os.environ['NEURONCORE_GROUP_SIZES'] = "1"
    user_workers = os.environ.get('SAGEMAKER_MODEL_SERVER_WORKERS')
    if user_workers is None:
        num_host_cores = os.environ.get("NEURON_CORE_HOST_TOTAL")
        if num_host_cores is None:
            os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = "1"
        else:
            os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = num_host_cores
    print("NEURONCORE_GROUP_SIZES {}".format(os.environ.get('NEURONCORE_GROUP_SIZES')))
    print("SAGEMAKER_MODEL_SERVER_WORKERS {}".format(os.environ.get('SAGEMAKER_MODEL_SERVER_WORKERS')))
    from sagemaker_mxnet_serving_container import serving
    serving.main()
else:
    subprocess.check_call(shlex.split(' '.join(sys.argv[1:])))

# prevent docker exit
subprocess.call(['tail', '-f', '/dev/null'])
