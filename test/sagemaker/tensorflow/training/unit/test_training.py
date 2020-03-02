# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import sys

from mock import MagicMock, patch
import pytest
from sagemaker_containers.beta.framework import runner
import tensorflow as tf

from sagemaker_tensorflow_container import training

MODULE_DIR = 's3://my/bucket'
MODULE_NAME = 'script_name'
LOG_LEVEL = 'Debug'
HOST1 = 'host1'
HOST2 = 'host2'
HOST_LIST = [HOST1, HOST2]
CURRENT_HOST = HOST1
CMD_ARGS = {'some_key': 'some_value'}
CLUSTER_WITH_PS = {
    'master': ['{}:2222'.format(HOST1)],
    'worker': ['{}:2222'.format(HOST2)],
    'ps': ['{}:2223'.format(HOST1), '{}:2223'.format(HOST2)]
}
MASTER_TASK = {'index': 0, 'type': 'master'}
WORKER_TASK = {'index': 0, 'type': 'worker'}
PS_TASK_1 = {'index': 0, 'type': 'ps'}
PS_TASK_2 = {'index': 1, 'type': 'ps'}
MODEL_DIR = 's3://bucket/prefix'
MODEL_DIR_CMD_LIST = ['--model_dir', MODEL_DIR]
REGION = 'us-west-2'
RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'resources')


@pytest.fixture
def distributed_training_env():
    env = simple_training_env()

    env.hosts = HOST_LIST
    env.additional_framework_parameters = {
        training.SAGEMAKER_PARAMETER_SERVER_ENABLED: True
    }
    return env


@pytest.fixture
def single_machine_training_env():
    return simple_training_env()


def simple_training_env():
    env = MagicMock()
    env.module_dir = MODULE_DIR
    env.user_entry_point = MODULE_NAME
    env.hyperparameters = {'model_dir': MODEL_DIR}
    env.log_level = LOG_LEVEL
    env.additional_framework_parameters = {}
    env.hosts = CURRENT_HOST
    env.current_host = CURRENT_HOST
    env.to_env_vars = lambda: {}
    env.job_name = 'test-training-job'
    return env


def test_is_host_master():
    assert training._is_host_master(HOST_LIST, CURRENT_HOST) is True
    assert training._is_host_master(HOST_LIST, 'host2') is False
    assert training._is_host_master(HOST_LIST, 'somehost') is False


@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_single_machine(run_module, single_machine_training_env):
    training.train(single_machine_training_env, MODEL_DIR_CMD_LIST)
    run_module.assert_called_with(MODULE_DIR, MODULE_NAME, MODEL_DIR_CMD_LIST,
                                  single_machine_training_env.to_env_vars(),
                                  runner=runner.ProcessRunnerType)


@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_horovod(run_module, single_machine_training_env):
    single_machine_training_env.additional_framework_parameters['sagemaker_mpi_enabled'] = True

    training.train(single_machine_training_env, MODEL_DIR_CMD_LIST)
    run_module.assert_called_with(MODULE_DIR, MODULE_NAME, MODEL_DIR_CMD_LIST,
                                  single_machine_training_env.to_env_vars(),
                                  runner=runner.MPIRunnerType)


@pytest.mark.skipif(sys.version_info.major != 3,
                    reason="Skip this for python 2 because of dict key order mismatch")
@patch('tensorflow.train.ClusterSpec')
@patch('tensorflow.train.Server')
@patch('sagemaker_containers.beta.framework.entry_point.run')
@patch('multiprocessing.Process', lambda target: target())
@patch('time.sleep', MagicMock())
def test_train_distributed_master(run, tf_server, cluster_spec, distributed_training_env):
    training.train(distributed_training_env, MODEL_DIR_CMD_LIST)

    cluster_spec.assert_called_with({'worker': ['host2:2222'],
                                     'master': ['host1:2222'],
                                     'ps': ['host1:2223', 'host2:2223']})

    tf_server.assert_called_with(
        cluster_spec(), job_name='ps', task_index=0, config=tf.ConfigProto(device_count={'GPU': 0})
    )
    tf_server().join.assert_called_with()

    tf_config = '{"cluster": {' \
                '"master": ["host1:2222"], ' \
                '"ps": ["host1:2223", "host2:2223"], ' \
                '"worker": ["host2:2222"]}, ' \
                '"environment": "cloud", ' \
                '"task": {"index": 0, "type": "master"}}'

    run.assert_called_with('s3://my/bucket', 'script_name', MODEL_DIR_CMD_LIST,
                           {'TF_CONFIG': tf_config})


@pytest.mark.skipif(sys.version_info.major != 3,
                    reason="Skip this for python 2 because of dict key order mismatch")
@patch('tensorflow.train.ClusterSpec')
@patch('tensorflow.train.Server')
@patch('sagemaker_containers.beta.framework.entry_point.run')
@patch('multiprocessing.Process', lambda target: target())
@patch('time.sleep', MagicMock())
def test_train_distributed_worker(run, tf_server, cluster_spec, distributed_training_env):
    distributed_training_env.current_host = HOST2

    training.train(distributed_training_env, MODEL_DIR_CMD_LIST)

    cluster_spec.assert_called_with({'worker': ['host2:2222'],
                                     'master': ['host1:2222'],
                                     'ps': ['host1:2223', 'host2:2223']})

    tf_server.assert_called_with(
        cluster_spec(), job_name='ps', task_index=1, config=tf.ConfigProto(device_count={'GPU': 0})
    )
    tf_server().join.assert_called_with()

    tf_config = '{"cluster": {' \
                '"master": ["host1:2222"], ' \
                '"ps": ["host1:2223", "host2:2223"], ' \
                '"worker": ["host2:2222"]}, ' \
                '"environment": "cloud", ' \
                '"task": {"index": 0, "type": "worker"}}'

    run.assert_called_with('s3://my/bucket', 'script_name', MODEL_DIR_CMD_LIST,
                           {'TF_CONFIG': tf_config})


@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_distributed_no_ps(run, distributed_training_env):
    distributed_training_env.additional_framework_parameters[
        training.SAGEMAKER_PARAMETER_SERVER_ENABLED] = False
    distributed_training_env.current_host = HOST2
    training.train(distributed_training_env, MODEL_DIR_CMD_LIST)

    run.assert_called_with(MODULE_DIR, MODULE_NAME, MODEL_DIR_CMD_LIST,
                           distributed_training_env.to_env_vars(), runner=runner.ProcessRunnerType)


def test_build_tf_config():
    assert training._build_tf_config(HOST_LIST, HOST1) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': MASTER_TASK
    }
    assert training._build_tf_config(HOST_LIST, HOST1, ps_task=True) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': PS_TASK_1
    }
    assert training._build_tf_config(HOST_LIST, HOST2) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': WORKER_TASK
    }
    assert training._build_tf_config(HOST_LIST, HOST2, ps_task=True) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': PS_TASK_2}


def test_build_tf_config_error():
    with pytest.raises(ValueError) as error:
        training._build_tf_config([HOST1], HOST1, ps_task=True)
    assert 'Cannot have a ps task if there are no parameter servers in the cluster' in str(error.value)


@patch('sagemaker_tensorflow_container.training.logger')
def test_log_model_missing_warning_no_model(logger):
    path = os.path.join(RESOURCE_PATH, 'test_dir_empty')
    if not os.path.exists(path):
        os.mkdir(path)
    training._log_model_missing_warning(path)
    logger.warn.assert_called_with('No model artifact is saved under path {}.'
                                   ' Your training job will not save any model files to S3.\n'
                                   'For details of how to construct your training script see:\n'
                                   'https://sagemaker.readthedocs.io/en/stable/using_tf.html#adapting-your-local-tensorflow-script'  # noqa
                                   .format(path))


@patch('sagemaker_tensorflow_container.training.logger')
def test_log_model_missing_warning_wrong_format(logger):
    training._log_model_missing_warning(os.path.join(RESOURCE_PATH, 'test_dir_wrong_model'))
    logger.warn.assert_called_with('Your model will NOT be servable with SageMaker TensorFlow Serving container. '
                                   'The model artifact was not saved in the TensorFlow '
                                   'SavedModel directory structure:\n'
                                   'https://www.tensorflow.org/guide/saved_model#structure_of_a_savedmodel_directory')


@patch('sagemaker_tensorflow_container.training.logger')
def test_log_model_missing_warning_wrong_parent_dir(logger):
    training._log_model_missing_warning(os.path.join(RESOURCE_PATH, 'test_dir_wrong_parent_dir'))
    logger.warn.assert_called_with('Your model will NOT be servable with SageMaker TensorFlow Serving containers. '
                                   'The SavedModel bundle is under directory \"{}\", not a numeric name.'
                                   .format('not-digit'))


@patch('sagemaker_tensorflow_container.training.logger')
def test_log_model_missing_warning_correct(logger):
    training._log_model_missing_warning(os.path.join(RESOURCE_PATH, 'test_dir_correct_model'))
    logger.warn.assert_not_called()


@patch('sagemaker_tensorflow_container.training.logger')
@patch('sagemaker_tensorflow_container.training.train')
@patch('logging.Logger.setLevel')
@patch('sagemaker_containers.beta.framework.training_env')
@patch('sagemaker_containers.beta.framework.env.read_hyperparameters', return_value={})
@patch('sagemaker_tensorflow_container.s3_utils.configure')
def test_main(configure_s3_env, read_hyperparameters, training_env,
              set_level, train, logger, single_machine_training_env):
    training_env.return_value = single_machine_training_env
    os.environ['SAGEMAKER_REGION'] = REGION
    training.main()
    read_hyperparameters.assert_called_once_with()
    training_env.assert_called_once_with(hyperparameters={})
    train.assert_called_once_with(single_machine_training_env, MODEL_DIR_CMD_LIST)
    configure_s3_env.assert_called_once()


@patch('sagemaker_tensorflow_container.training.logger')
@patch('sagemaker_tensorflow_container.training.train')
@patch('logging.Logger.setLevel')
@patch('sagemaker_containers.beta.framework.training_env')
@patch('sagemaker_containers.beta.framework.env.read_hyperparameters', return_value={'model_dir': MODEL_DIR})
@patch('sagemaker_tensorflow_container.s3_utils.configure')
def test_main_simple_training_model_dir(configure_s3_env, read_hyperparameters, training_env,
                                        set_level, train, logger, single_machine_training_env):
    training_env.return_value = single_machine_training_env
    os.environ['SAGEMAKER_REGION'] = REGION
    training.main()
    configure_s3_env.assert_called_once_with(MODEL_DIR, REGION)


@patch('sagemaker_tensorflow_container.training.logger')
@patch('sagemaker_tensorflow_container.training.train')
@patch('logging.Logger.setLevel')
@patch('sagemaker_containers.beta.framework.training_env')
@patch('sagemaker_containers.beta.framework.env.read_hyperparameters', return_value={'model_dir': MODEL_DIR,
                                                                                     '_tuning_objective_metric': 'auc'})
@patch('sagemaker_tensorflow_container.s3_utils.configure')
def test_main_tuning_model_dir(configure_s3_env, read_hyperparameters, training_env,
                               set_level, train, logger, single_machine_training_env):
    training_env.return_value = single_machine_training_env
    os.environ['SAGEMAKER_REGION'] = REGION
    training.main()
    expected_model_dir = '{}/{}/model'.format(MODEL_DIR, single_machine_training_env.job_name)
    configure_s3_env.assert_called_once_with(expected_model_dir, REGION)


@patch('sagemaker_tensorflow_container.training.logger')
@patch('sagemaker_tensorflow_container.training.train')
@patch('logging.Logger.setLevel')
@patch('sagemaker_containers.beta.framework.training_env')
@patch('sagemaker_containers.beta.framework.env.read_hyperparameters', return_value={'model_dir': '/opt/ml/model',
                                                                                     '_tuning_objective_metric': 'auc'})
@patch('sagemaker_tensorflow_container.s3_utils.configure')
def test_main_tuning_mpi_model_dir(configure_s3_env, read_hyperparameters, training_env,
                                   set_level, train, logger, single_machine_training_env):
    training_env.return_value = single_machine_training_env
    os.environ['SAGEMAKER_REGION'] = REGION
    training.main()
    configure_s3_env.assert_called_once_with('/opt/ml/model', REGION)
