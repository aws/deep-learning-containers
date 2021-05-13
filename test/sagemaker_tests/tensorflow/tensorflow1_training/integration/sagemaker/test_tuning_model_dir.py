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

import os

import pytest

from sagemaker.tuner import HyperparameterTuner, IntegerParameter

from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401
from .... import invoke_tensorflow_estimator


@pytest.mark.integration("hpo")
@pytest.mark.model("N/A")
def test_model_dir_with_training_job_name(sagemaker_session, n_virginia_sagemaker_session, ecr_image, n_virginia_ecr_image, instance_type, framework_version, multi_region_support):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    script = os.path.join(resource_path, 'tuning_model_dir', 'entry.py')

    estimator_parameter = {
            'entry_point': script,
            'role': 'SageMakerRole',
            'instance_type': instance_type,
            'instance_count': 1,
            'framework_version': framework_version,
            'py_version': 'py3'
            }
    estimator = invoke_tensorflow_estimator(ecr_image, n_virginia_ecr_image, sagemaker_session, n_virginia_sagemaker_session, estimator_parameter, multi_region_support)

    tuner = HyperparameterTuner(estimator=estimator,
                                objective_metric_name='accuracy',
                                hyperparameter_ranges={'arbitrary_value': IntegerParameter(0, 1)},
                                metric_definitions=[{'Name': 'accuracy', 'Regex': 'accuracy=([01])'}],
                                max_jobs=1,
                                max_parallel_jobs=1)

    # User script has logic to check for the correct model_dir
    tuner.fit(job_name=unique_name_from_base('test-tf-model-dir', max_length=32))
    tuner.wait()
