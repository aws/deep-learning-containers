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

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)
parser.add_argument('--arbitrary_value', type=int, default=0)
args = parser.parse_args()

assert os.environ['TRAINING_JOB_NAME'] in args.model_dir, 'model_dir not unique to training job: %s' % args.model_dir

# For the "hyperparameter tuning" to work
print('accuracy=1')
