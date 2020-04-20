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

import os
import time

import pytest
from sagemaker import utils
from sagemaker.tensorflow import TensorFlow
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent

from test.integration import DEFAULT_TIMEOUT
from test.integration import RESOURCE_PATH
from timeout import timeout

DATA_PATH = os.path.join(RESOURCE_PATH, "mnist")
SCRIPT_PATH = os.path.join(DATA_PATH, "mnist_gluon_basic_hook_demo.py")


@pytest.mark.skip_py2_containers
def test_training(sagemaker_session, ecr_image, instance_type, framework_version):

    sm_client = sagemaker_session.sagemaker_client

    experiment_name = "tf-container-integ-test-{}".format(int(time.time()))

    experiment = Experiment.create(
        experiment_name=experiment_name,
        description="Integration test experiment from sagemaker-tf-container",
        sagemaker_boto_client=sm_client,
    )

    trial_name = "tf-container-integ-test-{}".format(int(time.time()))

    trial = Trial.create(
        experiment_name=experiment_name, trial_name=trial_name, sagemaker_boto_client=sm_client
    )

    training_job_name = utils.unique_name_from_base("test-tf-experiments-mnist")

    # create a training job and wait for it to complete
    with timeout(minutes=DEFAULT_TIMEOUT):
        resource_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
        script = os.path.join(resource_path, "mnist", "mnist.py")
        estimator = TensorFlow(
            entry_point=script,
            role="SageMakerRole",
            train_instance_type=instance_type,
            train_instance_count=1,
            sagemaker_session=sagemaker_session,
            image_name=ecr_image,
            framework_version=framework_version,
            script_mode=True,
        )
        inputs = estimator.sagemaker_session.upload_data(
            path=os.path.join(resource_path, "mnist", "data"), key_prefix="scriptmode/mnist"
        )
        estimator.fit(inputs, job_name=training_job_name)

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
