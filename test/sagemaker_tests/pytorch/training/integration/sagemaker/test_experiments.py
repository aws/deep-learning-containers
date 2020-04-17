# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import time

import pytest
from sagemaker.pytorch import PyTorch
from sagemaker import utils
from ...integration import training_dir, smdebug_mnist_script, DEFAULT_TIMEOUT
from ...integration.sagemaker.timeout import timeout


@pytest.mark.skip_py2_containers
def test_training(sagemaker_session, ecr_image, instance_type):

    from smexperiments.experiment import Experiment
    from smexperiments.trial import Trial
    from smexperiments.trial_component import TrialComponent

    sm_client = sagemaker_session.sagemaker_client

    experiment_name = "pytorch-container-integ-test-{}".format(int(time.time()))

    experiment = Experiment.create(
        experiment_name=experiment_name,
        description="Integration test full customer e2e from sagemaker-pytorch-container",
        sagemaker_boto_client=sm_client,
    )

    trial_name = "pytorch-container-integ-test-{}".format(int(time.time()))
    trial = Trial.create(
        experiment_name=experiment_name, trial_name=trial_name, sagemaker_boto_client=sm_client
    )

    hyperparameters = {
        "random_seed": True,
        "num_steps": 50,
        "smdebug_path": "/opt/ml/output/tensors",
        "epochs": 1,
        "data_dir": training_dir,
    }

    training_job_name = utils.unique_name_from_base("test-pytorch-experiments-image")

    # create a training job and wait for it to complete
    with timeout(minutes=DEFAULT_TIMEOUT):
        pytorch = PyTorch(
            entry_point=smdebug_mnist_script,
            role="SageMakerRole",
            train_instance_count=1,
            train_instance_type=instance_type,
            sagemaker_session=sagemaker_session,
            image_name=ecr_image,
            hyperparameters=hyperparameters,
        )
        training_input = pytorch.sagemaker_session.upload_data(
            path=training_dir, key_prefix="pytorch/mnist"
        )
        pytorch.fit({"training": training_input}, job_name=training_job_name)

    training_job = sm_client.describe_training_job(TrainingJobName=training_job_name)
    training_job_arn = training_job["TrainingJobArn"]

    # verify trial component auto created from the training job
    trial_components = list(
        TrialComponent.list(source_arn=training_job_arn, sagemaker_boto_client=sm_client)
    )

    trial_component_summary = trial_components[0]
    trial_component = TrialComponent.load(
        trial_component_name=trial_component_summary.trial_component_name,
        sagemaker_boto_client=sm_client,
    )

    # associate the trial component with the trial
    trial.add_trial_component(trial_component)

    # cleanup
    trial.remove_trial_component(trial_component_summary.trial_component_name)
    trial_component.delete()
    trial.delete()
    experiment.delete()
