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
from sagemaker.train.configs import SourceCode, Compute

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT, mnist_path, throughput_path
from .timeout import timeout
from .test_distributed_operations import can_run_smmodelparallel
from ....training import get_efa_test_instance_type
from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag
from . import skip_if_not_v3_compatible, invoke_pytorch_model_trainer


def validate_or_skip_smdataparallel(ecr_image):
    if not can_run_smdataparallel(ecr_image):
        pytest.skip("Data Parallelism is supported on CUDA 11 on PyTorch v1.6 and above")


def can_run_smdataparallel(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.6") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("110")


def skip_unsupported_instances_smdataparallel(instance_type):
    if instance_type.startswith("ml.p5"):
        pytest.skip(f"{instance_type} is not supported by smdataparallel")


def validate_or_skip_smdataparallel_efa(ecr_image):
    if not can_run_smdataparallel_efa(ecr_image):
        pytest.skip("EFA is only supported on CUDA 11, and on PyTorch 1.8.1 or higher")


def can_run_smdataparallel_efa(ecr_image):
    _, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    return Version(image_framework_version) in SpecifierSet(">=1.8.1") and Version(
        image_cuda_version.strip("cu")
    ) >= Version("110")


@pytest.mark.skip("SMDDP binary releases are decoupled from DLC releases")
@pytest.mark.skip_cpu
@pytest.mark.skip_trcomp_containers
@pytest.mark.processor("gpu")
@pytest.mark.model("N/A")
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.team("smdataparallel")
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p4d.24xlarge"]), indirect=True
)
@pytest.mark.efa()
def test_smdataparallel_throughput(
    framework_version, ecr_image, sagemaker_regions, efa_instance_type, tmpdir
):
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smdataparallel_efa(ecr_image)
    skip_unsupported_instances_smdataparallel(efa_instance_type)

    source_code = SourceCode(
        source_dir=throughput_path,
        entry_script="smdataparallel_throughput.py",
    )
    compute_params = {"instance_type": efa_instance_type, "instance_count": 2}
    hyperparameters = {
        "size": 64,
        "num_tensors": 20,
        "iterations": 100,
        "warmup": 10,
        "bucket_size": 25,
        "info": f"PT-{efa_instance_type}-N2",
    }

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            hyperparameters=hyperparameters,
            job_name="test-pt-v3-smddp-throughput",
        )


@pytest.mark.skip("SMDDP binary releases are decoupled from DLC releases")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.processor("gpu")
@pytest.mark.team("smdataparallel")
def test_smdataparallel_mnist_script_mode_multigpu(
    ecr_image, sagemaker_regions, instance_type, tmpdir
):
    """
    Tests SM Distributed DataParallel single-node via script mode
    """
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smdataparallel(ecr_image)
    instance_type = "ml.p4d.24xlarge"

    source_code = SourceCode(
        source_dir=mnist_path,
        entry_script="smdataparallel_mnist_script_mode.sh",
    )
    compute_params = {"instance_type": instance_type, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-smddp-mnist-script-mode",
        )


@pytest.mark.skip("SMDDP binary releases are decoupled from DLC releases")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.flaky(reruns=2)
@pytest.mark.efa()
@pytest.mark.team("smdataparallel")
@pytest.mark.parametrize(
    "efa_instance_type",
    get_efa_test_instance_type(default=["ml.p4d.24xlarge"]),
    indirect=True,
)
def test_smdataparallel_mnist(ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    """
    Tests smddprun command via ModelTrainer distribution parameter
    """
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smdataparallel_efa(ecr_image)
    skip_unsupported_instances_smdataparallel(efa_instance_type)

    source_code = SourceCode(
        source_dir=mnist_path,
        entry_script="smdataparallel_mnist.py",
    )
    compute_params = {"instance_type": efa_instance_type, "instance_count": 2}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-smddp-mnist",
        )


@pytest.mark.skip("SMDDP binary releases are decoupled from DLC releases")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_trcomp_containers
@pytest.mark.processor("gpu")
@pytest.mark.skip_cpu
@pytest.mark.multinode(2)
@pytest.mark.integration("smdataparallel")
@pytest.mark.model("mnist")
@pytest.mark.flaky(reruns=2)
@pytest.mark.efa()
@pytest.mark.parametrize(
    "efa_instance_type", get_efa_test_instance_type(default=["ml.p4d.24xlarge"]), indirect=True
)
@pytest.mark.team("smdataparallel")
def test_hc_smdataparallel_mnist(ecr_image, sagemaker_regions, efa_instance_type, tmpdir):
    """
    Tests smddprun command via ModelTrainer distribution parameter
    """
    skip_if_not_v3_compatible(ecr_image)
    validate_or_skip_smdataparallel_efa(ecr_image)
    skip_unsupported_instances_smdataparallel(efa_instance_type)

    source_code = SourceCode(
        source_dir=mnist_path,
        entry_script="smdataparallel_mnist.py",
    )
    compute_params = {"instance_type": efa_instance_type, "instance_count": 2}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-hc-smddp-mnist",
        )


@pytest.mark.skip(
    "SMDDP binary releases are decoupled from DLC releases and SM Model Parallel team is maintaining their own Docker Container"
)
@pytest.mark.skip_cpu
@pytest.mark.skip_trcomp_containers
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.processor("gpu")
@pytest.mark.integration("smdataparallel_smmodelparallel")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("instance_types", ["ml.p4d.24xlarge"])
@pytest.mark.team("smdataparallel")
def test_smmodelparallel_smdataparallel_mnist(
    instance_types, ecr_image, sagemaker_regions, py_version, tmpdir
):
    """
    Tests SM Distributed DataParallel and ModelParallel single-node via script mode
    """
    skip_if_not_v3_compatible(ecr_image)
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

    source_code = SourceCode(
        source_dir=mnist_path,
        entry_script=entry_point,
    )
    compute_params = {"instance_type": instance_types, "instance_count": 1}

    with timeout(minutes=DEFAULT_TIMEOUT):
        invoke_pytorch_model_trainer(
            ecr_image,
            sagemaker_regions,
            source_code=source_code,
            compute_params=compute_params,
            job_name="test-pt-v3-smdmp-smddp-mnist",
        )
