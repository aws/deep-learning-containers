# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

from mock import MagicMock, patch

from sagemaker_tensorflow_container import s3_utils


BUCKET_REGION = 'us-west-2'
JOB_REGION = 'us-west-1'
JOB_BUKCET = 'sagemaker-us-west-2-000-00-1'
PREFIX = 'sagemaker/something'
MODEL_DIR = 's3://{}/{}'.format(JOB_BUKCET, PREFIX)


@patch('boto3.client')
def test_configure(client):
    s3 = MagicMock()
    client.return_value = s3
    loc = {'LocationConstraint': BUCKET_REGION}
    s3.get_bucket_location.return_value = loc

    s3_utils.configure(MODEL_DIR, JOB_REGION)

    assert os.environ['S3_REGION'] == BUCKET_REGION
    assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '1'
    assert os.environ['S3_USE_HTTPS'] == '1'


def test_configure_local_dir():
    s3_utils.configure('/opt/ml/model', JOB_REGION)

    assert os.environ['S3_REGION'] == JOB_REGION
    assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '1'
    assert os.environ['S3_USE_HTTPS'] == '1'
