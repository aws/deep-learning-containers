# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import shutil
import uuid

import pytest
from recordio_utils import build_record_file, build_single_record_file
from sagemaker import s3_input
from sagemaker.tensorflow import TensorFlow

from test.integration.utils import processor, py_version, unique_name_from_base  # noqa: F401
from timeout import timeout

DIMENSION = 5


def make_test_data(directory, name, num_files, num_records, dimension, sagemaker_session):
    if not os.path.exists('test-data'):
        os.makedirs('test-data')
    for i in range(num_files):
        if num_records > 1:
            build_record_file(os.path.join(directory, name + str(i)),
                              num_records=num_records, dimension=dimension)
        else:
            build_single_record_file(os.path.join(directory, name + str(i)),
                                     dimension=dimension)

    return sagemaker_session.upload_data(path=os.path.join(directory),
                                         key_prefix='pipemode-{}-files'.format(name))


@pytest.fixture(scope='session')
def multi_records_test_data(sagemaker_session):
    test_data = 'test-data-' + str(uuid.uuid4())
    os.makedirs(test_data)
    s3_url = make_test_data(
        directory=test_data,
        name='multi',
        num_files=1,
        num_records=1000,
        dimension=DIMENSION,
        sagemaker_session=sagemaker_session)
    shutil.rmtree(test_data)
    return s3_url


@pytest.fixture(scope='session')
def single_record_test_data(sagemaker_session):
    test_data = 'test-data-' + str(uuid.uuid4())
    os.makedirs(test_data)
    s3_url = make_test_data(
        directory=test_data,
        name='single',
        num_files=100,
        num_records=1,
        dimension=DIMENSION,
        sagemaker_session=sagemaker_session)
    shutil.rmtree(test_data)
    return s3_url


def run_test(sagemaker_session, ecr_image, instance_type, framework_version, test_data,
             record_wrapper_type=None):
    source_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'pipemode')
    script = os.path.join(source_path, 'pipemode.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           train_instance_type=instance_type,
                           train_instance_count=1,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           framework_version=framework_version,
                           script_mode=True,
                           input_mode='Pipe',
                           hyperparameters={'dimension': DIMENSION})
    input = s3_input(s3_data=test_data,
                     distribution='FullyReplicated',
                     record_wrapping=record_wrapper_type,
                     input_mode='Pipe')
    with timeout(minutes=20):
        estimator.fit({'elizabeth': input},
                      job_name=unique_name_from_base('test-sagemaker-pipemode'))


def test_single_record(sagemaker_session, ecr_image, instance_type, framework_version,
                       single_record_test_data):
    run_test(sagemaker_session,
             ecr_image,
             instance_type,
             framework_version,
             single_record_test_data,
             'RecordIO')


def test_multi_records(sagemaker_session, ecr_image, instance_type, framework_version,
                       multi_records_test_data):
    run_test(sagemaker_session,
             ecr_image,
             instance_type,
             framework_version,
             multi_records_test_data)
