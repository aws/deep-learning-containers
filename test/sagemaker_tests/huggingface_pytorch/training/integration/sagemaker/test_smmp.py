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
from ...integration import (DEFAULT_TIMEOUT)
from sagemaker.huggingface import HuggingFace
from ...integration.sagemaker.timeout import timeout

@pytest.mark.processor("gpu")
@pytest.mark.model("smmp")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_smmp_gpu(sagemaker_session, framework_version, ecr_image, instance_type, dist_gpu_backend):

    with timeout(minutes=DEFAULT_TIMEOUT):
        # hyperparameters, which are passed into the training job
        hyperparameters = {
            'model_name_or_path': 'roberta-large',
            'task_name': 'mnli',
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'do_train': True,
            'do_eval': True,
            'do_predict': True,
            'output_dir': '/opt/ml/model',
            'max_steps': 500,
        }

        # configuration for running training on smdistributed Model Parallel
        mpi_options = {
            "enabled": True,
            "processes_per_host": 8,
        }
        smp_options = {
            "enabled": True,
            "parameters": {
                "microbatches": 4,
                "placement_strategy": "spread",
                "pipeline": "interleaved",
                "optimize": "speed",
                "partitions": 4,
                "ddp": True,
            }
        }

        distribution = {
            "smdistributed": {"modelparallel": smp_options},
            "mpi": mpi_options
        }

        # instance configurations
        instance_type = instance_type or 'ml.p3dn.16xlarge'
        instance_count = 1
        volume_size = 400

        # git configuration to download our fine-tuning script
        git_config = {'repo': 'https://github.com/huggingface/notebooks.git', 'branch': 'master'}
        # metric definition to extract the results
        metric_definitions = [
            {'Name': 'train_runtime', 'Regex': "train_runtime.*=\D*(.*?)$"},
            {'Name': 'train_samples_per_second', 'Regex': "train_samples_per_second.*=\D*(.*?)$"},
            {'Name': 'epoch', 'Regex': "epoch.*=\D*(.*?)$"},
            {'Name': 'f1', 'Regex': "f1.*=\D*(.*?)$"},
            {'Name': 'exact_match', 'Regex': "exact_match.*=\D*(.*?)$"}]

        huggingface_estimator = HuggingFace(entry_point='run_glue.py',
                                            source_dir='./sagemaker/04_distributed_training_model_parallelism/scripts/',
                                            git_config=git_config,
                                            metrics_definition=metric_definitions,
                                            instance_type=instance_type,
                                            instance_count=instance_count,
                                            volume_size=volume_size,
                                            role='SageMakerRole',
                                            image_uri=ecr_image,
                                            framework_version='1.6.0',
                                            transformers_version='4.4.2',
                                            pytorch_version='1.6.0',
                                            py_version='py36',
                                            distribution=distribution,
                                            hyperparameters=hyperparameters,
                                            debugger_hook_config=False)
        huggingface_estimator.fit()
