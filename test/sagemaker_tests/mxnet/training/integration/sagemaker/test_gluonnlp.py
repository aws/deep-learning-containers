# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from ...integration import RESOURCE_PATH
from .... import invoke_mxnet_helper_function

NLP_DATA_PATH = os.path.join(RESOURCE_PATH, 'nlp')
NLP_SCRIPT_PATH = os.path.join(NLP_DATA_PATH, 'word_embedding.py')


@pytest.mark.integration("gluonnlp")
@pytest.mark.model("word_embeddings")
@pytest.mark.skip_py2_containers
def test_nlp_training(sagemaker_regions, ecr_image, instance_type, framework_version):
    estimator_parameters = {
        'entry_point': NLP_SCRIPT_PATH,
        'instance_count': 1,
        'instance_type': instance_type,
        'framework_version': framework_version,
        'train_max_run': 5 * 60
    }

    invoke_mxnet_helper_function(ecr_image, sagemaker_regions, _test_nlp_training, estimator_parameters)


def _test_nlp_training(ecr_image, sagemaker_session, **kwargs):
    nlp = MXNet(role='SageMakerRole',
                sagemaker_session=sagemaker_session,
                image_uri=ecr_image,
                **kwargs)

    job_name = utils.unique_name_from_base('test-nlp-image')
    nlp.fit(job_name=job_name)
