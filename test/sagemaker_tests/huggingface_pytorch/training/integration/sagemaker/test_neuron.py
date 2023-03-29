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
import sagemaker
from sagemaker import utils
from sagemaker.huggingface import HuggingFace
from ..... import invoke_sm_helper_function
from ...integration import DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
from retrying import retry

# configuration for running training on torch distributed Data Parallel
distribution = {"torch_distributed": {"enabled": True}}

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


def get_optimum_neuron_version(ecr_image):
    optimum_neuron_version_search = re.search(r"optimum(\d+(\.\d+){1,2})", ecr_image)
    if optimum_neuron_version_search:
        optimum_neuron_version = transformers_version_search.group(1)
        return optimum_neuron_version
    else:
        raise LookupError("HF optimum-neuron version not found in image URI")

@pytest.mark.processor("neuron")
@pytest.mark.model("hf-pt-qa-neuron")
@pytest.mark.neuron_test
@pytest.mark.skip_py2_containers
def test_neuron_question_answering(ecr_image, sagemaker_regions, py_version, instance_type):
    function_args = {
        'py_version': py_version,
        'instance_type': instance_type,
        'instance_count': 1,
        'num_neuron_cores': 2,
    }
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_neuron_question_answering_function, function_args)

def _test_neuron_question_answering_function(ecr_image, sagemaker_session, py_version, instance_type= "ml.trn1.32xlarge", instance_count=1, num_neuron_cores=2):
    optimum_neuron_version = get_optimum_neuron_version(ecr_image)
    git_config = {'repo': 'https://github.com/huggingface/optimum-neuron.git', 'branch': 'v' + optimum_neuron_version}

    source_dir = "./examples/question-answering"

    hyperparameters = {**hyperparameters, 'nproc-per-node': num_neuron_cores, 'nnodes': instance_count}

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
            # distribution=distribution,  # Uncomment when it is enabled by HuggingFace Estimator
            hyperparameters=hyperparameters,
        )
        estimator.fit(job_name=sagemaker.utils.unique_name_from_base('test-hf-pt-qa-neuron'))