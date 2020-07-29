# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

import pytest

from sagemaker import utils
from sagemaker.mxnet.estimator import MXNet
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner

from ...integration import RESOURCE_PATH
from .timeout import timeout

DATA_PATH = os.path.join(RESOURCE_PATH, 'mnist')
SCRIPT_PATH = os.path.join(DATA_PATH, 'mnist.py')


@pytest.mark.integration("hpo")
@pytest.mark.model("mnist")
def test_tuning(sagemaker_session, ecr_image, instance_type):
    mx = MXNet(entry_point=SCRIPT_PATH,
               role='SageMakerRole',
               train_instance_count=1,
               train_instance_type=instance_type,
               sagemaker_session=sagemaker_session,
               image_name=ecr_image,
               hyperparameters={'epochs': 1})

    hyperparameter_ranges = {'learning-rate': ContinuousParameter(0.01, 0.2)}
    objective_metric_name = 'Validation-accuracy'
    metric_definitions = [
        {'Name': 'Validation-accuracy', 'Regex': 'Validation-accuracy=([0-9\\.]+)'}]

    tuner = HyperparameterTuner(mx,
                                objective_metric_name,
                                hyperparameter_ranges,
                                metric_definitions,
                                max_jobs=2,
                                max_parallel_jobs=2)

    with timeout(minutes=20):
        prefix = 'mxnet_mnist/{}'.format(utils.sagemaker_timestamp())
        train_input = mx.sagemaker_session.upload_data(path=os.path.join(DATA_PATH, 'train'),
                                                       key_prefix=prefix + '/train')
        test_input = mx.sagemaker_session.upload_data(path=os.path.join(DATA_PATH, 'test'),
                                                      key_prefix=prefix + '/test')

        job_name = utils.unique_name_from_base('test-mxnet-image', max_length=32)
        tuner.fit({'train': train_input, 'test': test_input}, job_name=job_name)
        tuner.wait()
