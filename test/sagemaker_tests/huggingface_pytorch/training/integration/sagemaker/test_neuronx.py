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
import re
import pytest
import sagemaker
from sagemaker import utils
from sagemaker.huggingface import HuggingFace
from ...integration import DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
from retrying import retry
from sagemaker import get_execution_role

# configuration for running training on torch distributed Data Parallel
distribution = {"torch_distributed": {"enabled": True}}

# hyperparameters, which are passed into the training job
hyperparameters = {
    'model_name_or_path': 'bert-large-uncased-whole-word-masking',
    'dataset_name': 'squad',
    'do_train': True,
    'do_eval': True,
    'bf16': True,
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

# ValueError: Must setup local AWS configuration with a region supported by SageMaker.
def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)

# TBD. This function is mainly there to handle capacity issues now. Once trn1 capacaity issues
# are fixed, we can remove this function
@retry(
    stop_max_attempt_number=360, wait_fixed=10000, retry_on_exception=retry_if_value_error,
)
def invoke_neuron_helper_function(ecr_image, sagemaker_regions, helper_function, helper_function_args):
    """
    Used to invoke SM job defined in the helper functions in respective test file. The ECR image and the sagemaker
    session are passed explicitly depending on the AWS region.
    This function will rerun for all SM regions after a defined wait time if capacity issues are seen.

    :param ecr_image: ECR image in us-west-2 region
    :param sagemaker_regions: List of SageMaker regions
    :param helper_function: Function to invoke
    :param helper_function_args: Helper function args

    :return: None
    """
    from ..... import get_ecr_image_region, get_sagemaker_session, get_ecr_image

    ecr_image_region = get_ecr_image_region(ecr_image)
    for region in sagemaker_regions:
        sagemaker_session = get_sagemaker_session(region)
        # Reupload the image to test region if needed
        tested_ecr_image = get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        try:
            helper_function(tested_ecr_image, sagemaker_session, **helper_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                raise ValueError("CapacityError: Retry.")
            else:
                raise e

@pytest.mark.processor("neuronx")
@pytest.mark.model("hf-pt-qa-neuronx")
@pytest.mark.neuronx_test
@pytest.mark.skip_py2_containers
def test_neuronx_question_answering(ecr_image, sagemaker_regions, py_version, instance_type):
    function_args = {
        'py_version': py_version,
        'instance_type': instance_type,
        'instance_count': 1,
        'num_neuron_cores': 2,
    }
    invoke_neuron_helper_function(ecr_image, sagemaker_regions, _test_neuronx_question_answering_function, function_args)

def _test_neuronx_question_answering_function(ecr_image, sagemaker_session, py_version, instance_type= "ml.trn1.32xlarge", instance_count=1, num_neuron_cores=2):
    optimum_neuron_version = "0.0.3"
    git_config = {'repo': 'https://github.com/huggingface/optimum-neuron.git', 'branch': 'v' + optimum_neuron_version}

    source_dir = "./examples/question-answering"

    role = get_execution_role()
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
        estimator.fit(job_name=sagemaker.utils.unique_name_from_base('test-hf-pt-qa-neuronx'))