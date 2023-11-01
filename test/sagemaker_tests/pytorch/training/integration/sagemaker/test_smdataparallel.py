# Copyright 2018-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
from sagemaker import utils
from sagemaker.instance_group import InstanceGroup
from sagemaker.pytorch import PyTorch

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT, mnist_path, throughput_path
from ...integration.sagemaker.timeout import timeout
from ...integration.sagemaker.test_distributed_operations import (
    can_run_smmodelparallel,
    _disable_sm_profiler,
)
from ....training import get_efa_test_instance_type
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from . import invoke_pytorch_estimator


def validate_or_skip_smdataparallel(ecr_image):
    if not can_run_smdataparallel(ecr_image):
        pytest.skip("Data Parallelism is supported on CUDA 11 on PyTorch v1.6 and above")


def can_run_smdataparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.6") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("110")


def validate_or_skip_smdataparallel_efa(ecr_image):
    if not can_run_smdataparallel_efa(ecr_image):
        pytest.skip("EFA is only supported on CUDA 11, and on PyTorch 1.8.1 or higher")


def can_run_smdataparallel_efa(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.8.1") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("110")


@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.team("smdataparallel")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p4d.24xlarge"]), indirect=True
)
@pytest.mark.skip_cpu
@pytest.mark.skip_trcomp_containers
@pytest.mark.efa()
@pytest.mark.skip_pt21_test
@pytest.mark.skip_pt20_cuda121_tests
def test_smdataparallel_throughput(
    framework_version, ecr_image, sagemaker_regions, efa_instance_type, tmpdir
):
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_smdataparallel_efa(ecr_image)
        hyperparameters = {
            "size": 64,
            "num_tensors": 20,
            "iterations": 100,
            "warmup": 10,
            "bucket_size": 25,
            "info": f"PT-{efa_instance_type}-N2",
        }
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}
        estimator_parameter = {
            "entry_point": "smdataparallel_throughput.py",
            "role": "SageMakerRole",
            "instance_count": 2,
            "instance_type": efa_instance_type,
            "source_dir": throughput_path,
            "framework_version": framework_version,
            "hyperparameters": hyperparameters,
            "distribution": distribution,
        }

        job_name_prefix = "test-pt-smddp-throughput"
        invoke_pytorch_estimator(
            ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name_prefix
        )


@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.skip_pt21_test
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.team("smdataparallel")
def test_smdataparallel_mnist_script_mode_multigpu(
    ecr_image, sagemaker_regions, instance_type, tmpdir
):
    """
    Tests SM Distributed DataParallel single-node via script mode
    """
    validate_or_skip_smdataparallel(ecr_image)
    instance_type = "ml.p3.16xlarge"
    distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}
    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameter = {
            "entry_point": "smdataparallel_mnist_script_mode.sh",
            "role": "SageMakerRole",
            "source_dir": mnist_path,
            "instance_count": 1,
            "instance_type": instance_type,
            "distribution": distribution,
        }
        job_name_prefix = "test-pt-smddp-mnist-script-mode"
        invoke_pytorch_estimator(
            ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name_prefix
        )


@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
@pytest.mark.flaky(reruns=2)
@pytest.mark.efa()
@pytest.mark.skip_trcomp_containers
@pytest.mark.skip_pt21_test
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.team("smdataparallel")
@pytest.mark.parametrize(
    "efa_instance_type",
    get_efa_test_instance_type(default=["ml.p3.16xlarge", "ml.p4d.24xlarge"]),
    indirect=True,
)
def test_smdataparallel_mnist(ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    """
    Tests smddprun command via Estimator API distribution parameter
    """
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_smdataparallel_efa(ecr_image)
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}
        estimator_parameter = {
            "entry_point": "smdataparallel_mnist.py",
            "role": "SageMakerRole",
            "source_dir": mnist_path,
            "instance_count": 2,
            "instance_type": efa_instance_type,
            "distribution": distribution,
        }

        job_name_prefix = "test-pt-smddp-mnist"
        invoke_pytorch_estimator(
            ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name_prefix
        )


@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.skip_py2_containers
@pytest.mark.flaky(reruns=2)
@pytest.mark.efa()
@pytest.mark.skip_trcomp_containers
@pytest.mark.skip_pt21_test
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p3.16xlarge"]), indirect=True
)
@pytest.mark.team("smdataparallel")
def test_hc_smdataparallel_mnist(ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    """
    Tests smddprun command via Estimator API distribution parameter
    """
    with timeout(minutes=DEFAULT_TIMEOUT):
        validate_or_skip_smdataparallel_efa(ecr_image)
        instance_count = 2
        training_group = InstanceGroup("train_group", efa_instance_type, instance_count)
        distribution = {
            "smdistributed": {"dataparallel": {"enabled": True}},
            "instance_groups": [training_group],
        }
        estimator_parameter = {
            "entry_point": "smdataparallel_mnist.py",
            "role": "SageMakerRole",
            "source_dir": mnist_path,
            "instance_groups": [training_group],
            "distribution": distribution,
        }

        job_name_prefix = "test-pt-hc-smddp-mnist"
        invoke_pytorch_estimator(
            ecr_image, sagemaker_regions, estimator_parameter, job_name=job_name_prefix
        )


@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.skip_trcomp_containers
@pytest.mark.skip_pt21_test
@pytest.mark.integration("smdataparallel_smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("instance_types", ["ml.p3.16xlarge"])
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.team("smdataparallel")
def test_smmodelparallel_smdataparallel_mnist(
    instance_types, ecr_image, sagemaker_regions, py_version, tmpdir
):
    """
    Tests SM Distributed DataParallel and ModelParallel single-node via script mode
    This test has been added for SM DataParallelism and ModelParallelism tests for re:invent.
    TODO: Consider reworking these tests after re:Invent releases are done
    """
    can_run_modelparallel = can_run_smmodelparallel(ecr_image)
    can_run_dataparallel = can_run_smdataparallel(ecr_image)
    if can_run_dataparallel and can_run_modelparallel:
        entry_point = "smdataparallel_smmodelparallel_mnist_script_mode.sh"
    elif can_run_dataparallel:
        entry_point = "smdataparallel_mnist_script_mode.sh"
    elif can_run_modelparallel:
        entry_point = "smmodelparallel_mnist_script_mode.sh"
    else:
        pytest.skip("Both modelparallel and dataparallel dont support this image, nothing to run")

    with timeout(minutes=DEFAULT_TIMEOUT):
        estimator_parameter = {
            "entry_point": entry_point,
            "role": "SageMakerRole",
            "source_dir": mnist_path,
            "instance_count": 1,
            "instance_type": instance_types,
        }
        job_name_prefix = "test-pt-smdmp-smddp-mnist"
        invoke_pytorch_estimator(
            ecr_image,
            sagemaker_regions,
            estimator_parameter,
            disable_sm_profiler=True,
            job_name=job_name_prefix,
        )
