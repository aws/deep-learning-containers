# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace

from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
import sagemaker
import re

# configuration for running training on smdistributed Data Parallel
distribution = {'smdistributed': {'dataparallel': {'enabled': True}}}

# hyperparameters, which are passed into the training job
hyperparameters = {
    'model_name_or_path': 'bert-large-uncased-whole-word-masking',
    'dataset_name': 'squad',
    'do_train': True,
    'do_eval': True,
    'fp16': True,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'num_train_epochs': 1,
    'max_seq_length': 384,
    'max_steps': 10,
    'pad_to_max_length': True,
    'doc_stride': 128,
    'output_dir': '/opt/ml/model'
}
# metric definition to extract the results
metric_definitions = [
    {"Name": "train_runtime", "Regex": "train_runtime.*=\D*(.*?)$"},
    {'Name': 'train_samples_per_second', 'Regex': "train_samples_per_second.*=\D*(.*?)$"},
    {'Name': 'epoch', 'Regex': "epoch.*=\D*(.*?)$"},
    {'Name': 'f1', 'Regex': "f1.*=\D*(.*?)$"},
    {'Name': 'exact_match', 'Regex': "exact_match.*=\D*(.*?)$"}]


def validate_or_skip_smdataparallel(ecr_image):
    if not can_run_smdataparallel(ecr_image):
        pytest.skip("Data Parallelism is supported on CUDA 11 on PyTorch v1.6 and above")


def can_run_smdataparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.6") and Version(image_cuda_version.strip("cu")) >= Version("110")

def get_transformers_version(ecr_image):
    transformers_version_search = re.search(r"transformers(\d+(\.\d+){1,2})", ecr_image)
    if transformers_version_search:
        transformers_version = transformers_version_search.group(1)
        return transformers_version
    else:
        raise LookupError("HF transformers version not found in image URI")

@pytest.mark.integration("smdataparallel")
@pytest.mark.model("hf_qa_smdp")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
def test_smdp_question_answering(ecr_image, instance_type, py_version, sagemaker_session, tmpdir):
    """
    Tests SM Distributed DataParallel single-node via script mode
    """
    transformers_version = get_transformers_version(ecr_image)
    git_config = {'repo': 'https://github.com/huggingface/transformers.git', 'branch': 'v'+transformers_version}

    validate_or_skip_smdataparallel(ecr_image)
    instance_count = 1
    instance_type = "ml.p3.16xlarge"
    
    source_dir = (
        "./examples/question-answering"
        if Version(transformers_version) < Version("4.6")
        else "./examples/pytorch/question-answering"
    )
    
    
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator = HuggingFace(
            entry_point='run_qa.py',
            source_dir=source_dir,
            git_config=git_config,
            metric_definitions=metric_definitions,
            role='SageMakerRole',
            image_uri=ecr_image,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            py_version=py_version,
            distribution=distribution,
            hyperparameters=hyperparameters,
        )
        estimator.fit(job_name=sagemaker.utils.unique_name_from_base('test-hf-pt-qa-smdp'))


@pytest.mark.integration("smdataparallel")
@pytest.mark.model("hf_qa_smdp_multi")
@pytest.mark.multinode(2)
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
def test_smdp_question_answering_multinode(ecr_image, instance_type, py_version, sagemaker_session, tmpdir):
    """
    Tests SM Distributed DataParallel single-node via script mode
    """

    transformers_version = get_transformers_version(ecr_image)
    git_config = {'repo': 'https://github.com/huggingface/transformers.git', 'branch': 'v'+transformers_version}

    validate_or_skip_smdataparallel(ecr_image)

    instance_count = 2
    instance_type = "ml.p3.16xlarge"
    
    
    source_dir = (
        "./examples/question-answering"
        if Version(transformers_version) < Version("4.6")
        else "./examples/pytorch/question-answering"
    )
    
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator = HuggingFace(
            entry_point='run_qa.py',
            source_dir=source_dir,
            git_config=git_config,
            metric_definitions=metric_definitions,
            role='SageMakerRole',
            image_uri=ecr_image,
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            py_version=py_version,
            distribution=distribution,
            hyperparameters=hyperparameters,
        )
        estimator.fit(job_name=sagemaker.utils.unique_name_from_base('test-hf-pt-qa-smdp-multi'))
