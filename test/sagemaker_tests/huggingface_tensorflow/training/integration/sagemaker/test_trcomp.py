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
from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig

from packaging.version import Version

from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag

import unittest.mock as mock


RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
BERT_PATH = os.path.join(RESOURCE_PATH, "scripts")

hyperparameters = {
    "max_steps": 3,
    "train_batch_size": 16,
    "model_name": "distilbert-base-uncased",
}


@pytest.mark.integration("sagmaker-training-compiler")
@pytest.mark.processor("gpu")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_huggingface_containers
@pytest.mark.skip_cpu
@mock.patch('sagemaker.huggingface.TrainingCompilerConfig.validate', return_value=None)
class TestSingleNodeSingleGPU:
    '''
    All Single Node Single GPU tests go here.
    '''
    @pytest.mark.model("distilbert-base")
    def test_trcomp_default(self, patched, sagemaker_session, ecr_image, tmpdir, capsys):
        '''
        Tests the default configuration of SM trcomp
        '''
        instance_type = "ml.p3.2xlarge"
        instance_count = 1

        estimator = HuggingFace(
            compiler_config=TrainingCompilerConfig(),
            entry_point="train.py",
            source_dir=BERT_PATH,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            image_uri=ecr_image,
            py_version=py_version,
            sagemaker_session=sagemaker_session,
            hyperparameters=hyperparameters,
            debugger_hook_config=False,  # currently needed
            max_retry_attempts=15,
        )

        estimator.fit(job_name=unique_name_from_base("hf-tf-trcomp-single-gpu-default"), logs=True)
        
        captured = capsys.readouterr()
        logs = captured.out+captured.err
        assert "Found configuration for Training Compiler" in logs


    @pytest.mark.model("distilbert-base")
    def test_trcomp_enabled(self, patched, sagemaker_session, ecr_image, tmpdir, capsys):
        '''
        Tests the explicit enabled configuration of SM trcomp
        '''
        instance_type = "ml.p3.2xlarge"
        instance_count = 1

        estimator = HuggingFace(
            compiler_config=TrainingCompilerConfig(enabled=True),
            entry_point="train.py",
            source_dir=BERT_PATH,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            image_uri=ecr_image,
            py_version=py_version,
            sagemaker_session=sagemaker_session,
            hyperparameters=hyperparameters,
            debugger_hook_config=False,  # currently needed
            max_retry_attempts=15,
        )

        estimator.fit(job_name=unique_name_from_base("hf-tf-trcomp-single-gpu-enabled"), logs=True)
        
        captured = capsys.readouterr()
        logs = captured.out+captured.err
        assert "Found configuration for Training Compiler" in logs


    @pytest.mark.model("distilbert-base")
    def test_trcomp_debug(self, patched, sagemaker_session, ecr_image, tmpdir, capsys):
        '''
        Tests the debug mode configuration of SM trcomp
        '''
        instance_type = "ml.p3.2xlarge"
        instance_count = 1

        estimator = HuggingFace(
            compiler_config=TrainingCompilerConfig(debug=True),
            entry_point="train.py",
            source_dir=BERT_PATH,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            image_uri=ecr_image,
            py_version=py_version,
            sagemaker_session=sagemaker_session,
            hyperparameters=hyperparameters,
            debugger_hook_config=False,  # currently needed
            max_retry_attempts=15,
        )

        estimator.fit(job_name=unique_name_from_base("hf-tf-trcomp-single-gpu-debug"), logs=True)
        
        captured = capsys.readouterr()
        logs = captured.out+captured.err
        assert "Found configuration for Training Compiler" in logs
        assert "Training Compiler set to debug mode" in logs


