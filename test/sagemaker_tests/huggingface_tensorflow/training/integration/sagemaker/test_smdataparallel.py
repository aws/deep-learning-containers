# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sagemaker
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace

from packaging.version import Version

from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
BERT_PATH = os.path.join(RESOURCE_PATH, 'scripts')

# hyperparameters, which are passed into the training job
hyperparameters={
    'max_steps': 10,
    'train_batch_size': 16,
    'model_name':'distilbert-base-uncased',
}

@pytest.mark.integration("hf_smdp")
@pytest.mark.model("hf_distilbert")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
# TODO: Enable sagemaker debugger, resolve github issue after enabling.
#  https://github.com/aws/deep-learning-containers/issues/1053
def test_hf_smdp(sagemaker_session, instance_type, ecr_image, tmpdir, framework_version):
    """
    Tests SMDataParallel single-node command via script mode
    """
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if Version(image_framework_version) < Version("2.3.1") or image_cuda_version != "cu110":
        pytest.skip("Data Parallelism is only supported on CUDA 11, and on TensorFlow 2.3.1 or higher")

    # configuration for running training on smdistributed Data Parallel
    distribution = {'smdistributed': {'dataparallel': {'enabled': True}}}

    instance_type = "ml.p3.16xlarge"
    instance_count = 1

    estimator = HuggingFace(
        entry_point='train.py',
        source_dir=BERT_PATH,
        role='SageMakerRole',
        instance_type=instance_type,
        instance_count=instance_count,
        image_uri=ecr_image,
        framework_version=framework_version,
        py_version=py_version,
        distribution=distribution,
        sagemaker_session=sagemaker_session,
        hyperparameters=hyperparameters,
        debugger_hook_config=False,  # currently needed
    )

    estimator.fit(job_name=unique_name_from_base('test-tf-hf-smdp'))


@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.multinode(2)
@pytest.mark.integration("hf_smdp_multinode")
@pytest.mark.model("hf_distilbert")
@pytest.mark.skip_py2_containers
@pytest.mark.parametrize('instance_types', ["ml.p3.16xlarge"])
# Skipping `ml.p3dn.24xlarge` instance type due to capacity issue in us-west-2
# TODO: Enable sagemaker debugger, resolve github issue after enabling.
#  https://github.com/aws/deep-learning-containers/issues/1053
def test_hf_smdp_multi(instance_types, ecr_image, py_version, sagemaker_session, tmpdir, framework_version):
    """
    Tests smddprun command via Estimator API distribution parameter
    """
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if Version(image_framework_version) < Version("2.3.1") or image_cuda_version != "cu110":
        pytest.skip("Data Parallelism is only supported on CUDA 11, and on TensorFlow 2.3.1 or higher")

    distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}
    instance_type = "ml.p3.16xlarge"
    instance_count = 2

    estimator = HuggingFace(entry_point='train.py',
                           source_dir=BERT_PATH,
                           role='SageMakerRole',
                           instance_type=instance_type,
                           instance_count=instance_count,
                           image_uri=ecr_image,
                           framework_version=framework_version,
                           py_version=py_version,
                           sagemaker_session=sagemaker_session,
                           hyperparameters=hyperparameters,
                           distribution=distribution,
                           debugger_hook_config=False,  # currently needed
                           )

    estimator.fit(job_name=unique_name_from_base('test-tf-hf-smdp-multi'))
