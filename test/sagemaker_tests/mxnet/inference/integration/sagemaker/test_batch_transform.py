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

import json
import os
from urllib.parse import urlparse

import pytest

from sagemaker import utils
from sagemaker.mxnet.model import MXNetModel

from ...integration import RESOURCE_PATH
from ...integration.sagemaker import timeout

SCRIPT_PATH = os.path.join(RESOURCE_PATH, 'default_handlers', 'model', 'code', 'empty_module.py')
MNIST_PATH = os.path.join(RESOURCE_PATH, 'mnist')
MODEL_PATH = os.path.join(MNIST_PATH, 'model', 'model.tar.gz')

DATA_FILE = '07.csv'
DATA_PATH = os.path.join(MNIST_PATH, 'images', DATA_FILE)


@pytest.mark.integration("batch_transform")
@pytest.mark.model("mnist")
def test_batch_transform(sagemaker_session, ecr_image, instance_type, framework_version):
    s3_prefix = 'mxnet-serving/mnist'
    model_data = sagemaker_session.upload_data(path=MODEL_PATH, key_prefix=s3_prefix)
    model = MXNetModel(model_data,
                       'SageMakerRole',
                       SCRIPT_PATH,
                       image=ecr_image,
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_session)

    transformer = model.transformer(1, instance_type)
    with timeout.timeout_and_delete_model_with_transformer(transformer, sagemaker_session, minutes=20):
        input_data = sagemaker_session.upload_data(path=DATA_PATH, key_prefix=s3_prefix)

        job_name = utils.unique_name_from_base('test-mxnet-serving-batch')
        transformer.transform(input_data, content_type='text/csv', job_name=job_name)
        transformer.wait()

    prediction = _transform_result(sagemaker_session.boto_session, transformer.output_path)
    assert prediction == 7


def _transform_result(boto_session, output_path):
    s3 = boto_session.resource('s3', region_name=boto_session.region_name)

    parsed_url = urlparse(output_path)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:]

    output_obj = s3.Object(bucket_name, '{}/{}.out'.format(prefix, DATA_FILE))
    output = output_obj.get()['Body'].read().decode('utf-8')

    probabilities = json.loads(output)[0]
    return probabilities.index(max(probabilities))
