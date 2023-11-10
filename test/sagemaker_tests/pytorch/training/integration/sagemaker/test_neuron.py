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

import pytest
import sagemaker
from sagemaker import utils
from sagemaker.pytorch import PyTorch
from ...integration import neuron_allreduce_path, neuron_mlp_path, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
from retrying import retry


def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)


# TBD. This function is mainly there to handle capacity issues now. Once trn1 capacaity issues
# are fixed, we can remove this function
@retry(
    stop_max_attempt_number=5,
    wait_fixed=60000,
    retry_on_exception=retry_if_value_error,
)
def invoke_neuron_helper_function(
    ecr_image, sagemaker_regions, helper_function, helper_function_args
):
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
        tested_ecr_image = (
            get_ecr_image(ecr_image, region) if region != ecr_image_region else ecr_image
        )
        try:
            helper_function(tested_ecr_image, sagemaker_session, **helper_function_args)
            return
        except sagemaker.exceptions.UnexpectedStatusException as e:
            if "CapacityError" in str(e):
                raise ValueError("CapacityError: Retry.")
            else:
                raise e


@pytest.mark.processor("neuronx")
@pytest.mark.model("unknown_model")
@pytest.mark.parametrize("instance_types", ["ml.trn1.32xlarge"])
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_allreduce_distributed(
    framework_version, ecr_image, sagemaker_regions, instance_types
):
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_types,
        "instance_count": 2,
    }
    invoke_neuron_helper_function(
        ecr_image, sagemaker_regions, _test_neuron_allreduce_distributed, function_args
    )


@pytest.mark.processor("neuronx")
@pytest.mark.model("mlp")
@pytest.mark.parametrize("instance_types", ["ml.trn1.32xlarge"])
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_mlp_distributed(framework_version, ecr_image, sagemaker_regions, instance_types):
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_types,
        "instance_count": 2,
    }
    invoke_neuron_helper_function(
        ecr_image, sagemaker_regions, _test_neuron_mlp_distributed, function_args
    )


@pytest.mark.processor("neuronx")
@pytest.mark.model("unknown_model")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_allreduce_process(framework_version, ecr_image, sagemaker_regions, instance_type):
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "num_neuron_cores": 2,
    }
    invoke_neuron_helper_function(
        ecr_image, sagemaker_regions, _test_neuron_allreduce, function_args
    )


@pytest.mark.processor("neuronx")
@pytest.mark.model("mlp")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
def test_neuron_mlp_process(framework_version, ecr_image, sagemaker_regions, instance_type):
    function_args = {
        "framework_version": framework_version,
        "instance_type": instance_type,
        "num_neuron_cores": 2,
    }
    invoke_neuron_helper_function(ecr_image, sagemaker_regions, _test_neuron_mlp, function_args)


def _test_neuron_allreduce(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type,
    instance_count=1,
    num_neuron_cores=2,
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point="entrypoint.py",
            source_dir=neuron_allreduce_path,
            role="SageMakerRole",
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={"nproc-per-node": num_neuron_cores, "nnodes": instance_count},
            disable_profiler=True,
        )

        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(
            path=neuron_allreduce_path, key_prefix="pytorch/neuron_allreduce"
        )

        pytorch.fit(
            {"required_argument": fake_input},
            job_name=utils.unique_name_from_base("test-pt-neuron-allreduce"),
        )


def _test_neuron_mlp(
    ecr_image,
    sagemaker_session,
    framework_version,
    instance_type,
    instance_count=1,
    num_neuron_cores=2,
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point="entrypoint.py",
            source_dir=neuron_mlp_path,
            role="SageMakerRole",
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={"nproc-per-node": num_neuron_cores, "nnodes": instance_count},
            disable_profiler=True,
        )

        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(
            path=neuron_mlp_path, key_prefix="pytorch/neuron_mlp"
        )

        pytorch.fit(
            {"required_argument": fake_input},
            job_name=utils.unique_name_from_base("test-pt-neuron-mlp"),
        )


def _test_neuron_allreduce_distributed(
    ecr_image, sagemaker_session, framework_version, instance_type, instance_count=1
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point="all_reduce.py",
            source_dir=neuron_allreduce_path,
            role="SageMakerRole",
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            distribution={"torch_distributed": {"enabled": True}},
            disable_profiler=True,
            environment={"FI_EFA_FORK_SAFE": "1"},
        )

        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(
            path=neuron_allreduce_path, key_prefix="pytorch/neuron_allreduce"
        )

        pytorch.fit(
            {"required_argument": fake_input},
            job_name=utils.unique_name_from_base("test-pt-neuron-allreduce-dist"),
        )


def _test_neuron_mlp_distributed(
    ecr_image, sagemaker_session, framework_version, instance_type, instance_count=1
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point="train_torchrun.py",
            source_dir=neuron_mlp_path,
            role="SageMakerRole",
            instance_count=instance_count,
            instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            distribution={"torch_distributed": {"enabled": True}},
            disable_profiler=True,
            environment={"FI_EFA_FORK_SAFE": "1"},
        )

        pytorch.sagemaker_session.default_bucket()
        fake_input = pytorch.sagemaker_session.upload_data(
            path=neuron_mlp_path, key_prefix="pytorch/neuron_mlp"
        )

        pytorch.fit(
            {"required_argument": fake_input},
            job_name=utils.unique_name_from_base("test-pt-neuron-mlp-dist"),
        )
