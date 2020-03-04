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
import tempfile

import pytest
from sagemaker import utils
from sagemaker.mxnet.model import MXNetModel

from ...integration import RESOURCE_PATH
from ...integration.sagemaker import timeout

GLUONNLP_PATH = os.path.join(RESOURCE_PATH, 'gluonnlp')
SCRIPT_PATH = os.path.join(GLUONNLP_PATH, 'bert.py')


@pytest.mark.skip_py2_containers
def test_gluonnlp(sagemaker_session, ecr_image, instance_type, framework_version):
    import urllib.request
    tmpdir = tempfile.mkdtemp()
    tmpfile = os.path.join(tmpdir, 'bert_sst.tar.gz')
    urllib.request.urlretrieve('https://aws-dlc-sample-models.s3.amazonaws.com/bert_sst/bert_sst.tar.gz', tmpfile)

    prefix = 'gluonnlp-serving/default-handlers'
    model_data = sagemaker_session.upload_data(path=tmpfile, key_prefix=prefix)

    model = MXNetModel(model_data,
                       'SageMakerRole',
                       SCRIPT_PATH,
                       image=ecr_image,
                       py_version="py3",
                       framework_version=framework_version,
                       sagemaker_session=sagemaker_session)

    endpoint_name = utils.unique_name_from_base('test-mxnet-gluonnlp')
    with timeout.timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session):
        predictor = model.deploy(1, instance_type, endpoint_name=endpoint_name)

        output = predictor.predict(["Positive sentiment", "Negative sentiment"])
        assert [1, 0] == output
