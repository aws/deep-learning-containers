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

import os, subprocess, tarfile

import pytest
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace, TrainingCompilerConfig

from test.test_utils import (
    get_framework_and_version_from_tag,
    get_cuda_version_from_tag,
)
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from ...integration import DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout
import sagemaker
import re

import unittest.mock as mock


hyperparameters = {
    "model_name_or_path": "bert-large-uncased-whole-word-masking",
    "dataset_name": "squad",
    "do_train": True,
    "do_eval": True,
    "fp16": True,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 1,
    "max_seq_length": 384,
    "max_steps": 3,
    "max_eval_samples": 10,
    "pad_to_max_length": True,
    "doc_stride": 128,
    "output_dir": "/opt/ml/model",
}
# metric definition to extract the results
metric_definitions = [
    {"Name": "train_runtime", "Regex": "'train_runtime':\D*([0-9,.]*?)"},
    {"Name": "device", "Regex": "Using\D*([a-zA-Z0-9:]*)\D*device"},
    {
        "Name": "train_samples_per_second",
        "Regex": "train_samples_per_second.*=\D*(.*?)$",
    },
    {"Name": "epoch", "Regex": "epoch.*=\D*(.*?)$"},
    {"Name": "f1", "Regex": "f1.*=\D*(.*?)$"},
    {"Name": "exact_match", "Regex": "exact_match.*=\D*(.*?)$"},
]


def get_transformers_version(ecr_image):
    transformers_version_search = re.search(r"transformers(\d+(\.\d+){1,2})", ecr_image)
    if transformers_version_search:
        transformers_version = transformers_version_search.group(1)
        return transformers_version
    else:
        raise LookupError("HF transformers version not found in image URI")


@pytest.fixture
def instance_type():
    return "ml.p3.2xlarge"


@pytest.fixture
def instance_count():
    return 1


@pytest.fixture
def num_gpus_per_instance(instance_type):
    if instance_type in ["ml.p3.16xlarge", "ml.p4d.24xlarge"]:
        return 8
    elif instance_type in ["ml.g4dn.12xlarge", "ml.g5.12xlarge"]:
        return 4
    raise NotImplementedError("Unforeseen Instance Type")


@pytest.fixture
def should_nccl_use_pcie(instance_type, instance_count, ecr_image):
    """Should NCCL be explicitly forced to use PCIE when NVLINK is not available ? This is baked in from PyTorch 1.12."""
    pytorch_version = get_framework_and_version_from_tag(ecr_image)[1]
    if "g" in instance_type and (Version(pytorch_version) in SpecifierSet("< 1.12")):
        return True
    return False


@pytest.mark.integration("sagmaker-training-compiler")
@pytest.mark.processor("gpu")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_huggingface_containers
@pytest.mark.skip_cpu
@mock.patch("sagemaker.huggingface.TrainingCompilerConfig.validate", return_value=None)
class TestSingleNodeSingleGPU:
    """
    All Single Node Single GPU tests go here.
    """

    @pytest.mark.model("bert-large")
    def test_trcomp_default(
        self,
        patched,
        ecr_image,
        sagemaker_session,
        tmpdir,
        py_version,
        capsys,
        instance_type,
        instance_count,
    ):
        """
        Tests the default configuration of SM trcomp
        """
        transformers_version = get_transformers_version(ecr_image)
        git_config = {
            "repo": "https://github.com/huggingface/transformers.git",
            "branch": "v" + transformers_version,
        }

        source_dir = (
            "./examples/question-answering"
            if Version(transformers_version) < Version("4.6")
            else "./examples/pytorch/question-answering"
        )

        with timeout(minutes=DEFAULT_TIMEOUT):
            estimator = HuggingFace(
                compiler_config=TrainingCompilerConfig(),
                entry_point="run_qa.py",
                source_dir=source_dir,
                git_config=git_config,
                metric_definitions=metric_definitions,
                role="SageMakerRole",
                image_uri=ecr_image,
                instance_count=instance_count,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                hyperparameters=hyperparameters,
                py_version=py_version,
                max_retry_attempts=15,
            )
            estimator.fit(
                job_name=sagemaker.utils.unique_name_from_base(
                    "hf-pt-trcomp-SNSG-default"
                ),
                logs=True,
            )
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert "Found configuration for Training Compiler" in logs
        assert "Configuring SM Training Compiler" in logs
        assert "device: xla" in logs

    @pytest.mark.model("bert-large")
    def test_trcomp_enabled(
        self,
        patched,
        ecr_image,
        sagemaker_session,
        tmpdir,
        py_version,
        capsys,
        instance_type,
        instance_count,
    ):
        """
        Tests the explicit enabled configuration of SM trcomp
        """
        transformers_version = get_transformers_version(ecr_image)
        git_config = {
            "repo": "https://github.com/huggingface/transformers.git",
            "branch": "v" + transformers_version,
        }

        source_dir = (
            "./examples/question-answering"
            if Version(transformers_version) < Version("4.6")
            else "./examples/pytorch/question-answering"
        )

        with timeout(minutes=DEFAULT_TIMEOUT):
            estimator = HuggingFace(
                compiler_config=TrainingCompilerConfig(enabled=True),
                entry_point="run_qa.py",
                source_dir=source_dir,
                git_config=git_config,
                metric_definitions=metric_definitions,
                role="SageMakerRole",
                image_uri=ecr_image,
                instance_count=instance_count,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                hyperparameters=hyperparameters,
                py_version=py_version,
                max_retry_attempts=15,
            )
            estimator.fit(
                job_name=sagemaker.utils.unique_name_from_base(
                    "hf-pt-trcomp-SNSG-enabled"
                ),
                logs=True,
            )
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert "Found configuration for Training Compiler" in logs
        assert "Configuring SM Training Compiler" in logs
        assert "device: xla" in logs

    @pytest.mark.model("bert-large")
    def test_trcomp_debug(
        self,
        patched,
        ecr_image,
        sagemaker_session,
        tmpdir,
        py_version,
        capsys,
        instance_type,
        instance_count,
    ):
        """
        Tests the debug mode configuration of SM trcomp
        """
        transformers_version = get_transformers_version(ecr_image)
        git_config = {
            "repo": "https://github.com/huggingface/transformers.git",
            "branch": "v" + transformers_version,
        }

        source_dir = (
            "./examples/question-answering"
            if Version(transformers_version) < Version("4.6")
            else "./examples/pytorch/question-answering"
        )

        with timeout(minutes=DEFAULT_TIMEOUT):
            estimator = HuggingFace(
                compiler_config=TrainingCompilerConfig(debug=True),
                entry_point="run_qa.py",
                source_dir=source_dir,
                git_config=git_config,
                metric_definitions=metric_definitions,
                role="SageMakerRole",
                image_uri=ecr_image,
                instance_count=instance_count,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                hyperparameters=hyperparameters,
                py_version=py_version,
                max_retry_attempts=15,
            )
            estimator.fit(
                job_name=sagemaker.utils.unique_name_from_base(
                    "hf-pt-trcomp-SNSG-debug"
                ),
                logs=True,
            )

        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert "Found configuration for Training Compiler" in logs
        assert "Training Compiler set to debug mode" in logs
        assert "Configuring SM Training Compiler" in logs
        assert "device: xla" in logs

        debug_artifact_path = estimator.model_data.replace(
            "model.tar.gz", "output.tar.gz"
        )
        debug_artifact = os.path.join(tmpdir, "output.tar.gz")
        subprocess.check_output(
            ["aws", "s3", "cp", debug_artifact_path, debug_artifact]
        )
        with tarfile.open(debug_artifact, "r:gz") as tarball:
            tarball.extractall(path=tmpdir)
        xla_metrics_file = os.path.join(tmpdir, "compiler", "XLA_METRICS_FILE.txt")
        assert os.path.exists(xla_metrics_file)


@pytest.mark.integration("sagmaker-training-compiler")
@pytest.mark.processor("gpu")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_huggingface_containers
@pytest.mark.skip_cpu
@mock.patch("sagemaker.huggingface.TrainingCompilerConfig.validate", return_value=None)
class TestSingleNodeMultiGPU:
    """
    All Single Node Multi GPU tests go here.
    """

    @pytest.mark.parametrize(
        "instance_type, instance_count",
        [
            ("ml.p3.16xlarge", 1),
            ("ml.g4dn.12xlarge", 1),
            ("ml.g5.12xlarge", 1),
        ],
    )
    @pytest.mark.model("bert-large")
    def test_trcomp_default(
        self,
        patched,
        ecr_image,
        sagemaker_session,
        tmpdir,
        py_version,
        capsys,
        instance_type,
        instance_count,
        num_gpus_per_instance,
        should_nccl_use_pcie,
    ):
        """
        Tests the default configuration of SM trcomp
        """
        transformers_version = get_transformers_version(ecr_image)
        git_config = {
            "repo": "https://github.com/huggingface/transformers.git",
            "branch": "v" + transformers_version,
        }

        source_dir = (
            "./examples/question-answering"
            if Version(transformers_version) < Version("4.6")
            else "./examples/pytorch/question-answering"
        )

        hyperparameters["max_steps"] = 3 * num_gpus_per_instance

        with timeout(minutes=DEFAULT_TIMEOUT):
            estimator = HuggingFace(
                compiler_config=TrainingCompilerConfig(),
                entry_point="run_qa.py",
                source_dir=source_dir,
                git_config=git_config,
                metric_definitions=metric_definitions,
                role="SageMakerRole",
                image_uri=ecr_image,
                instance_count=instance_count,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                hyperparameters=hyperparameters,
                py_version=py_version,
                max_retry_attempts=15,
                distribution={"pytorchxla": {"enabled": True}},
                environment={"NCCL_P2P_LEVEL": "PXB"}
                if should_nccl_use_pcie
                else {},  # Temporary measure to enable communication through PCIe instead of NVLink
            )
            estimator.fit(
                job_name=sagemaker.utils.unique_name_from_base(
                    "hf-pt-trcomp-SNMG-default"
                ),
                logs=True,
            )
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert "Found configuration for Training Compiler" in logs
        assert "Configuring SM Training Compiler" in logs
        assert "device: xla" in logs
        assert "Invoking PT-XLA Runner" in logs
        assert "distributed training through PT-XLA Runtime" in logs
        assert "torch_xla.distributed.xla_spawn" in logs
        assert f"nranks {num_gpus_per_instance}" in logs


@pytest.mark.integration("sagmaker-training-compiler")
@pytest.mark.processor("gpu")
@pytest.mark.skip_py2_containers
@pytest.mark.skip_huggingface_containers
@pytest.mark.skip_cpu
@mock.patch("sagemaker.huggingface.TrainingCompilerConfig.validate", return_value=None)
class TestMultiNodeMultiGPU:
    """
    All Multi Node Multi GPU tests go here.
    """

    @pytest.mark.parametrize(
        "instance_type, instance_count",
        [
            ("ml.p3.16xlarge", 2),
            ("ml.p4d.24xlarge", 2),
            ("ml.g4dn.12xlarge", 2),
            ("ml.g5.12xlarge", 2),
        ],
    )
    @pytest.mark.model("bert-large")
    def test_trcomp_default(
        self,
        patched,
        ecr_image,
        sagemaker_session,
        tmpdir,
        py_version,
        capsys,
        instance_type,
        instance_count,
        num_gpus_per_instance,
        should_nccl_use_pcie,
    ):
        """
        Tests the default configuration of SM trcomp
        """
        transformers_version = get_transformers_version(ecr_image)
        git_config = {
            "repo": "https://github.com/huggingface/transformers.git",
            "branch": "v" + transformers_version,
        }

        source_dir = (
            "./examples/question-answering"
            if Version(transformers_version) < Version("4.6")
            else "./examples/pytorch/question-answering"
        )

        total_gpus = num_gpus_per_instance * instance_count
        hyperparameters["max_steps"] = 3 * total_gpus

        with timeout(minutes=DEFAULT_TIMEOUT):
            estimator = HuggingFace(
                compiler_config=TrainingCompilerConfig(),
                entry_point="run_qa.py",
                source_dir=source_dir,
                git_config=git_config,
                metric_definitions=metric_definitions,
                role="SageMakerRole",
                image_uri=ecr_image,
                instance_count=instance_count,
                instance_type=instance_type,
                sagemaker_session=sagemaker_session,
                hyperparameters=hyperparameters,
                py_version=py_version,
                max_retry_attempts=15,
                distribution={"pytorchxla": {"enabled": True}},
                environment={"NCCL_P2P_LEVEL": "PXB"}
                if should_nccl_use_pcie
                else {},  # Temporary measure to enable communication through PCIe instead of NVLink
            )
            estimator.fit(
                job_name=sagemaker.utils.unique_name_from_base(
                    "hf-pt-trcomp-MNMG-default"
                ),
                logs=True,
            )
        captured = capsys.readouterr()
        logs = captured.out + captured.err
        assert "Found configuration for Training Compiler" in logs
        assert "Configuring SM Training Compiler" in logs
        assert "device: xla" in logs
        assert "Invoking PT-XLA Runner" in logs
        assert "distributed training through PT-XLA Runtime" in logs
        assert "torch_xla.distributed.xla_spawn" in logs
        assert f"nranks {total_gpus}" in logs
