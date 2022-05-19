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
from unittest import mock
import os

import boto3, sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.training_compiler.config import TrainingCompilerConfig



resource_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')



def _assert_file_exists_in_s3(region, s3_url):
    parsed_url = urlparse(s3_url)
    s3 = boto3.resource('s3', region_name=region)
    s3.Object(parsed_url.netloc, parsed_url.path.lstrip('/')).load()


def _assert_model_exported_to_s3(estimator):
    region = estimator.sagemaker_session.boto_region_name
    s3_url = estimator.model_data
    _assert_file_exists_in_s3(region, s3_url)


def _assert_checkpoints_exported_to_s3(estimator, checkpoint_number):
    region = estimator.sagemaker_session.boto_region_name
    model_dir = estimator.model_data
    _assert_file_exists_in_s3(region, os.path.join(model_dir, 'checkpoint'))
    _assert_file_exists_in_s3(region,
                           os.path.join(model_dir, 'model.ckpt-{}.index'.format(checkpoint_number)))



@pytest.fixture()
def mnist_dataset(sagemaker_session):
    inputs = sagemaker_session.upload_data(
                                    path=os.path.join(resource_path, 'mnist', 'data'),
                                    key_prefix='scriptmode/mnist')
    return inputs


@pytest.fixture()
def mnist_distributed_dataset(sagemaker_session):
    inputs = sagemaker_session.upload_data(
                                    path=os.path.join(resource_path, 'mnist', 'data-distributed'),
                                    key_prefix='scriptmode/mnist-distributed')
    return inputs



@pytest.mark.xfail
@pytest.mark.gpu_only
@pytest.mark.integration("trcomp")
class TestDistributedTraining:
    

    @pytest.fixture()
    def instance_type(self):
        return 'ml.p3.8xlarge'


    @pytest.fixture()
    def instance_count(self):
        return 2


    def test_native(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir, mnist_dataset):
        script = os.path.join(resource_path, 'mnist', 'mnist.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                               },
                               )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base('test-TF-trcomp-DT'))
        _assert_model_exported_to_s3(estimator)


    @pytest.mark.integration("parameter server")
    def test_parameter_server(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir, mnist_distributed_dataset):
        script = os.path.join(resource_path, 'mnist', 'mnist_custom.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                                    TensorFlow.LAUNCH_PS_ENV_NAME: True,
                               },
                               )
        estimator.fit(mnist_distributed_dataset, job_name=unique_name_from_base('test-TF-trcomp-DT-PS'))
        _assert_checkpoints_exported_to_s3(estimator, 10)


    @pytest.mark.integration("horovod")
    def test_horovod(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        script = os.path.join(resource_path, 'mnist', 'horovod_mnist.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                                    TensorFlow.LAUNCH_MPI_ENV_NAME: True,
                                    TensorFlow.MPI_CUSTOM_MPI_OPTIONS: "-verbose -x orte_base_help_aggregate=0",
                                    TensorFlow.MPI_NUM_PROCESSES_PER_HOST: 1,
                               },
                               )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-DT-horovod"))
        _assert_model_exported_to_s3(estimator)


    @pytest.mark.integration("smdataparallel")
    def test_smdp(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        script = os.path.join(resource_path, 'mnist', 'smdataparallel_mnist.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                               },
                               distribution={"smdistributed": {"dataparallel": {"enabled": True}}},
                               )
        estimator.fit(job_name=unique_name_from_base("test-TF-trcomp-DT-SMDP"))


    @pytest.mark.integration("smmodelparallel")
    def test_smmp(self, sagemaker_session, ecr_image, framework_version, efa_instance_type, instance_count, tmpdir):
        path = os.path.join(resource_path, 'smmodelparallel')
        estimator = TensorFlow(source_dir=path,
                               entry_point='tf2_conv.py',
                               role='SageMakerRole',
                               instance_type=efa_instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
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


    @pytest.mark.integration("horovod")
    @pytest.mark.integration("smmodelparallel")
    def test_smmp_with_horovod(self, sagemaker_session, ecr_image, framework_version, efa_instance_type, instance_count, tmpdir):
        path = os.path.join(resource_path, 'smmodelparallel')
        estimator = TensorFlow(source_dir=path,
                               entry_point='smmodelparallel_hvd2_conv_multinode.py',
                               role='SageMakerRole',
                               instance_type=efa_instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
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


@pytest.mark.gpu_only
@pytest.mark.integration("trcomp")
class TestMLWorkFlow:


    @pytest.fixture()
    def instance_type(self):
        return 'ml.p3.2xlarge'


    @pytest.fixture()
    def instance_count(self):
        return 1


    @pytest.mark.xfail
    @pytest.mark.skip(reason="skip the test temporarily due to timeout issue")
    @pytest.mark.integration("smdebug")
    def test_smdebugger(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir, mnist_dataset):
        script = os.path.join(resource_path, 'mnist', 'mnist_smdebug.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                                    'smdebug_path': '/tmp/ml/output/tensors',
                               },
                            )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base('test-TF-trcomp-debug'))
        _assert_model_exported_to_s3(estimator)


    def test_training(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        script = os.path.join(resource_path, 'mnist', 'mnist.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                               },
                               )
        estimator.fit(mnist_dataset, job_name=unique_name_from_base('test-TF-trcomp'))
        _assert_model_exported_to_s3(estimator)


    @pytest.mark.integration("s3 plugin")
    def test_s3_plugin(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir, mnist_distributed_dataset):
        script = os.path.join(resource_path, 'mnist', 'mnist_custom.py')
        estimator = TensorFlow(entry_point=script,
                               role='SageMakerRole',
                               instance_type=instance_type,
                               instance_count=instance_count,
                               sagemaker_session=sagemaker_session,
                               image_uri=ecr_image,
                               framework_version=framework_version,
                               hyperparameters={
                                   # Saving a checkpoint after every 5 steps to hammer the S3 plugin
                                   'save-checkpoint-steps': 10,
                                   # Reducing throttling for checkpoint and model saving
                                   'throttle-secs': 1,
                                   # Without the patch training jobs would fail around 100th to
                                   # 150th step
                                   'max-steps': 200,
                                   # Large batch size would result in a larger checkpoint file
                                   'batch-size': 1024,
                                   # This makes the training job exporting model during training.
                                   # Stale model garbage collection will also be performed.
                                   'export-model-during-training': True,
                                    TrainingCompilerConfig.HP_ENABLE_COMPILER : True,
                               },
                               )
        estimator.fit(mnist_distributed_dataset, job_name=unique_name_from_base('test-TF-trcomp-s3'))
        _assert_checkpoints_exported_to_s3(estimator, 10)


    @pytest.mark.xfail
    @pytest.mark.integration("hpo")
    def test_hyperparameter_tuner(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        raise NotImplementedError()


    @pytest.mark.xfail
    @pytest.mark.integration("spot")
    def test_spot_training(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        raise NotImplementedError()


    @pytest.mark.xfail
    @pytest.mark.integration("neo")
    def test_inference_compiler_neo(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        raise NotImplementedError()


    @pytest.mark.xfail
    @pytest.mark.integration("serving")
    def test_serving(self, sagemaker_session, ecr_image, framework_version, instance_type, instance_count, tmpdir):
        raise NotImplementedError()


