# Copyright 2017-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from six.moves.urllib.parse import urlparse
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import boto3
from sagemaker.tensorflow import TensorFlow
from sagemaker.training_compiler.config import TrainingCompilerConfig

from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401


resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources")


def _assert_training_compiler_invoked(captured):
    logs = captured.out + captured.err
    assert "XLA is active" in logs
    assert "Found configuration for Training Compiler" in logs


def _assert_fault_attributed_to_training_compiler(captured):
    logs = captured.out + captured.err
    assert "SMTrainingCompiler Error: " in logs


def _assert_file_exists_in_s3(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource("s3", region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip("/")).load()


def _assert_model_exported_to_s3(estimator):
    region = estimator.sagemaker_session.boto_region_name
    s3_url = estimator.model_data
    _assert_file_exists_in_s3(region, s3_url)


def _assert_checkpoints_exported_to_s3(estimator, checkpoint_number):
    region = estimator.sagemaker_session.boto_region_name
    model_dir = estimator.model_dir
    _assert_file_exists_in_s3(region, os.path.join(model_dir, "checkpoint"))
    _assert_file_exists_in_s3(
        region, os.path.join(model_dir, "model.ckpt-{}.index".format(checkpoint_number))
    )


@pytest.fixture()
def mnist_dataset(sagemaker_session):
    inputs = sagemaker_session.upload_data(
        path=os.path.join(resource_path, "mnist", "data"), key_prefix="scriptmode/mnist"
    )
    return inputs


@pytest.fixture()
def mnist_distributed_dataset(sagemaker_session):
    inputs = sagemaker_session.upload_data(
        path=os.path.join(resource_path, "mnist", "data-distributed"),
        key_prefix="scriptmode/mnist-distributed",
    )
    return inputs


@pytest.fixture(autouse=True)
def smtrcomp_only(framework_version, ecr_image, request):
    if Version(framework_version) >= Version("2.12.0"):
        pytest.skip("Training Compiler support is not added in TF 2.12.0 and above")
    if Version(framework_version) in SpecifierSet("<2.9.1"):
        pytest.skip("Training Compiler support was added with TF 2.9.1")
    if "gpu" not in ecr_image:
        pytest.skip("Training Compiler is only available for GPUs")


@pytest.mark.multinode(2)
@pytest.mark.integration("trcomp")
@pytest.mark.team("training-compiler")
class TestDistributedTraining:
    @pytest.fixture()
    def instance_type(self):
        return "ml.p3.8xlarge"

    @pytest.fixture()
    def instance_count(self):
        return 2

    @pytest.mark.model("toy")
    def test_native(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        mnist_dataset,
        capsys,
    ):
        script = os.path.join(resource_path, "mnist", "mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
            },
        )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base("test-TF-trcomp-DT"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.model("LeNet")
    @pytest.mark.integration("parameter server")
    def test_parameter_server(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        mnist_distributed_dataset,
        capsys,
    ):
        script = os.path.join(resource_path, "mnist", "mnist_custom.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                TensorFlow.LAUNCH_PS_ENV_NAME: True,
            },
        )
        estimator.fit(
            mnist_distributed_dataset, job_name=unique_name_from_base("test-TF-trcomp-DT-PS")
        )
        _assert_checkpoints_exported_to_s3(estimator, 10)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.skip(reason="Trcomp behavior with Horovod is undefined")
    @pytest.mark.model("toy")
    @pytest.mark.integration("horovod")
    def test_horovod(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        capsys,
    ):
        script = os.path.join(resource_path, "mnist", "horovod_mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                TensorFlow.LAUNCH_MPI_ENV_NAME: True,
                TensorFlow.MPI_CUSTOM_MPI_OPTIONS: "-verbose -x orte_base_help_aggregate=0",
                TensorFlow.MPI_NUM_PROCESSES_PER_HOST: 1,
            },
            environment={
                "HOROVOD_ENABLE_XLA_OPS": "1",
            },
        )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-DT-horovod"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.usefixtures("feature_smddp_present")
    @pytest.mark.skip(reason="Trcomp behavior with SMDP is undefined")
    @pytest.mark.model("toy")
    @pytest.mark.integration("smdataparallel")
    @pytest.mark.team("smdataparallel")
    def test_smdp(
        self, sagemaker_session, ecr_image, framework_version, instance_count, tmpdir, capsys
    ):
        script = os.path.join(resource_path, "mnist", "smdataparallel_mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type="ml.p3.16xlarge",
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
            },
            distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
        )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-DT-SMDP"))
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.usefixtures("feature_smmp_present")
    @pytest.mark.skip(
        reason="SMMP is only supported on CUDA 11 on TensorFlow version between v2.3.1(inclusive) and v2.7.0(exclusive)"
    )
    @pytest.mark.model("toy")
    @pytest.mark.team("smmodelparallel")
    @pytest.mark.integration("smmodelparallel")
    def test_smmp(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        efa_instance_type,
        instance_count,
        tmpdir,
        capsys,
    ):
        path = os.path.join(resource_path, "smmodelparallel")
        estimator = TensorFlow(
            source_dir=path,
            entry_point="tf2_conv.py",
            role="SageMakerRole",
            instance_type=efa_instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
            },
            distributions={
                "mpi": {
                    "enabled": True,
                    "processes_per_host": 2,
                    "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa ",
                }
            },
        )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-DT-SMMP"))
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.usefixtures("feature_smmp_present")
    @pytest.mark.skip(
        reason="SMMP is only supported on CUDA 11 on TensorFlow version between v2.3.1(inclusive) and v2.7.0(exclusive)"
    )
    @pytest.mark.model("toy")
    @pytest.mark.integration("horovod")
    @pytest.mark.team("smmodelparallel")
    @pytest.mark.integration("smmodelparallel")
    def test_smmp_with_horovod(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        efa_instance_type,
        instance_count,
        tmpdir,
        capsys,
    ):
        path = os.path.join(resource_path, "smmodelparallel")
        estimator = TensorFlow(
            source_dir=path,
            entry_point="smmodelparallel_hvd2_conv_multinode.py",
            role="SageMakerRole",
            instance_type=efa_instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
            },
            distributions={
                "mpi": {
                    "enabled": True,
                    "processes_per_host": 2,
                    "custom_mpi_options": "-verbose --mca orte_base_help_aggregate 0 -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_PROVIDER=efa ",
                }
            },
        )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-DT-SMMP-horovod"))
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)


@pytest.mark.integration("trcomp")
@pytest.mark.team("training-compiler")
class TestMLWorkFlow:
    @pytest.fixture()
    def instance_type(self):
        return "ml.p3.2xlarge"

    @pytest.fixture()
    def instance_count(self):
        return 1

    @pytest.mark.usefixtures("feature_smdebug_present")
    @pytest.mark.skip(reason="skip the test temporarily due to timeout issue")
    @pytest.mark.model("toy")
    @pytest.mark.integration("smdebug")
    def test_smdebugger(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        mnist_dataset,
        capsys,
    ):
        script = os.path.join(resource_path, "mnist", "mnist_smdebug.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                "smdebug_path": "/tmp/ml/output/tensors",
            },
        )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base("test-TF-trcomp-debug"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.model("toy")
    def test_training(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        mnist_dataset,
        capsys,
    ):
        script = os.path.join(resource_path, "mnist", "mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
            },
        )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base("test-TF-trcomp"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.skip("Skipping this test on TF DLCs")
    @pytest.mark.model("distilbert")
    def test_BYOC_training(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        capsys,
    ):
        source_path = os.path.join(resource_path, "mlm")
        estimator = TensorFlow(
            entry_point="run_mlm.py",
            source_dir=source_path,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            disable_profiler=True,
            debugger_hook_config=False,
            py_version="py38",
            volume_size=500,
            model_dir=False,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                "model_name_or_path": "distilbert-base-uncased",
                "max_seq_length": 128,
                "dataset_name": "wikitext",
                "dataset_config_name": "wikitext-2-raw-v1",
                "max_steps": 3,
                "fp16": 1,
                "num_train_epochs": 1,
                "per_device_train_batch_size": 160,
                "do_train": True,
                "do_eval": False,
                "overwrite_output_dir": True,
                "save_strategy": "no",
                "logging_strategy": "no",
                "evaluation_strategy": "no",
                "output_dir": "/opt/ml/model",
            },
        )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-BYOC"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.model("LeNet")
    @pytest.mark.integration("s3 plugin")
    def test_s3_plugin(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        mnist_distributed_dataset,
        capsys,
    ):
        script = os.path.join(resource_path, "mnist", "mnist_custom.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                # Saving a checkpoint after every 5 steps to hammer the S3 plugin
                "save-checkpoint-steps": 10,
                # Reducing throttling for checkpoint and model saving
                "throttle-secs": 1,
                # Without the patch training jobs would fail around 100th to
                # 150th step
                "max-steps": 200,
                # Large batch size would result in a larger checkpoint file
                "batch-size": 1024,
                # This makes the training job exporting model during training.
                # Stale model garbage collection will also be performed.
                "export-model-during-training": True,
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
            },
        )
        estimator.fit(
            mnist_distributed_dataset, job_name=unique_name_from_base("test-TF-trcomp-s3")
        )
        _assert_checkpoints_exported_to_s3(estimator, 10)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)

    @pytest.mark.skip(reason="SM Training Compiler team yet to implement this integration test")
    @pytest.mark.model("N/A")
    @pytest.mark.integration("hpo")
    def test_hyperparameter_tuner(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        capsys,
    ):
        raise NotImplementedError()

    @pytest.mark.model("toy")
    @pytest.mark.integration("serving")
    def test_serving(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        capsys,
        mnist_dataset,
    ):
        script = os.path.join(resource_path, "mnist", "mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                "save-as-tf": True,
            },
        )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base("test-TF-trcomp-serving"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            image_uri=f"763104351884.dkr.ecr.{estimator.sagemaker_session.boto_region_name}.amazonaws.com/tensorflow-inference:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker",
        )
        predictor.delete_predictor()

    @pytest.mark.skip(
        reason="Neo does not support TF 2.11 yet. TF 2.11's saved models are not backwards compatible."
    )
    @pytest.mark.model("toy")
    @pytest.mark.integration("neo")
    def test_inference_compiler_neo(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        capsys,
        mnist_dataset,
    ):
        script = os.path.join(resource_path, "mnist", "mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                "save-as-tf": True,
            },
        )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base("test-TF-trcomp-serving"))
        _assert_model_exported_to_s3(estimator)
        captured = capsys.readouterr()
        _assert_training_compiler_invoked(captured)
        s3_prefix = estimator.model_data.replace("output/model.tar.gz", "")
        estimator.compile_model(
            target_instance_family="ml_p3",
            input_shape={"data": [1, 28, 28]},
            output_path=s3_prefix,
            framework="tensorflow",
            framework_version="2.9",
        )


@pytest.mark.integration("trcomp")
@pytest.mark.team("training-compiler")
class TestUserExperience:
    @pytest.fixture()
    def instance_type(self):
        return "ml.p3.2xlarge"

    @pytest.fixture()
    def instance_count(self):
        return 1

    @pytest.mark.skip(
        reason="Signature based failure attribution does not differentiate between XLA errors and TF/Py errors. Stack trace based failure attribution does not work with TF since the stack trace is manipulate to abstract away graph mode execution details."
    )
    @pytest.mark.model("toy")
    @pytest.mark.integration("toolkit")
    def test_toolkit_fault_attribution(
        self,
        sagemaker_session,
        ecr_image,
        framework_version,
        instance_type,
        instance_count,
        tmpdir,
        capsys,
    ):
        script = os.path.join(resource_path, "smtrcomp", "tensorflow_issue_58135.py")
        estimator_args = dict(
            entry_point=script,
            role="SageMakerRole",
            instance_type=instance_type,
            instance_count=instance_count,
            sagemaker_session=sagemaker_session,
            image_uri=ecr_image,
            framework_version=framework_version,
        )
        native_estimator = TensorFlow(
            **estimator_args,
        )
        smtrcomp_estimator = TensorFlow(
            **estimator_args,
            hyperparameters={
                TrainingCompilerConfig.HP_ENABLE_COMPILER: True,
                "jit_compile": True,
            },
        )
        # If a training script+args is successful without SM Training Compiler
        native_estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-fault"))
        # But the training script+args fails with SM Training Compiler
        with pytest.raises(Exception):
            smtrcomp_estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-fault"))
        # Then the failure reason should clearly indicate that this failure is due to SM Training Compiler
        reason = sagemaker_session.describe_training_job(
            smtrcomp_estimator.latest_training_job.name
        )["FailureReason"]
        assert "SMTrainingCompiler Error" in reason
