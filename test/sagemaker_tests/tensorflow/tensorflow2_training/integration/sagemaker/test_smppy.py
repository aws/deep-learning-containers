# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os, sys
import subprocess

# only the latest version of sagemaker supports profiler
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker>=2.180.0"])

import time
from test.test_utils import get_framework_and_version_from_tag

import boto3
import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from sagemaker import ProfilerConfig, Profiler
from sagemaker.tensorflow import TensorFlow

from ..... import invoke_sm_helper_function
from ...integration.utils import processor, py_version, unique_name_from_base  # noqa: F401

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "resources")
MNIST_PATH = os.path.join(RESOURCE_PATH, "mnist")

INSTANCE_TYPE = "ml.g4dn.12xlarge"
WAIT_TIME = 60
NUM_CHECKS = 5


def _skip_if_image_is_not_compatible_with_smppy(image_uri):
    _, framework_version = get_framework_and_version_from_tag(image_uri)
    compatible_versions = SpecifierSet("==2.11.*")
    if Version(framework_version) not in compatible_versions:
        pytest.skip(f"This test only works for TF versions in {compatible_versions}")


@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy(ecr_image, sagemaker_regions, py_version, tmpdir):
    _skip_if_image_is_not_compatible_with_smppy(ecr_image)
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_smppy_mnist_function)


def _test_smppy_mnist_function(ecr_image, sagemaker_session):
    script = os.path.join(MNIST_PATH, "mnist_smppy.py")
    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        instance_count=1,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        image_uri=ecr_image,
        profiler_config=ProfilerConfig(profile_params=Profiler(cpu_profiling_duration=3600)),
        debug_hook_config=False,
    )

    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_PATH, "data"),
        key_prefix="scriptmode/mnist",
    )

    name = unique_name_from_base("test-tf-smppy-training")
    bucket = None

    try:
        estimator.fit(inputs, job_name=name)
        s3 = boto3.client("s3")
        bucket = estimator.output_path.replace("s3://", "").rstrip("/")

        # Give postprocessing rule some time to complete

        prefix = _get_deep_profiler_rule_output_prefix(estimator)
        postproc_contents = []
        checks = 0
        while not postproc_contents and checks < NUM_CHECKS:
            time.sleep(WAIT_TIME)
            postproc_contents = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents")
            checks += 1

        assert (
            len(postproc_contents) > 0
        ), f"The prefix {prefix} doesn't contain any sagemaker profiler files"
        for file in postproc_contents:
            assert file.get("Size") > 0, "sagemaker profiler file has size 0"

    finally:
        _cleanup_s3_output(bucket, name)


@pytest.mark.processor("gpu")
@pytest.mark.integration("smppy")
@pytest.mark.model("mnist")
@pytest.mark.multinode(2)
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_training_smppy_multinode(ecr_image, sagemaker_regions, py_version, tmpdir):
    _skip_if_image_is_not_compatible_with_smppy(ecr_image)
    invoke_sm_helper_function(ecr_image, sagemaker_regions, _test_smppy_mnist_multinode_function)


def _test_smppy_mnist_multinode_function(ecr_image, sagemaker_session):
    script = os.path.join(MNIST_PATH, "mnist_smppy.py")
    estimator = TensorFlow(
        entry_point=script,
        role="SageMakerRole",
        instance_count=2,
        instance_type=INSTANCE_TYPE,
        sagemaker_session=sagemaker_session,
        image_uri=ecr_image,
        profiler_config=ProfilerConfig(profile_params=Profiler(cpu_profiling_duration=3600)),
        debug_hook_config=False,
    )

    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(MNIST_PATH, "data"),
        key_prefix="scriptmode/mnist",
    )

    name = unique_name_from_base("test-tf-smppy-training-multinode")
    bucket = None
    prefix = ""

    try:
        estimator.fit(inputs, job_name=name)
        s3 = boto3.client("s3")
        bucket = estimator.output_path.replace("s3://", "").rstrip("/")

        # Give postprocessing rule some time to complete

        prefix = _get_deep_profiler_rule_output_prefix(estimator)
        postproc_contents = []
        checks = 0
        while not postproc_contents and checks < NUM_CHECKS:
            time.sleep(WAIT_TIME)
            postproc_contents = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents")
            checks += 1

        assert (
            len(postproc_contents) > 0
        ), f"The prefix {prefix} doesn't contain any sagemaker profiler files"
        for file in postproc_contents:
            assert file.get("Size") > 0, "sagemaker profiler file has size 0"

    finally:
        _cleanup_s3_output(bucket, name)


def _cleanup_s3_output(bucket, name):
    if bucket:
        s3 = boto3.client("s3")
        all_contents = s3.list_objects_v2(Bucket=bucket, Prefix=os.path.join(name, "")).get(
            "Contents"
        )
        for file in all_contents:
            s3.delete_object(Bucket=bucket, Key=file["Key"])
    else:
        print("Bucket name was not set")


def _get_deep_profiler_rule_output_prefix(estimator):
    config_name = None
    for processing in estimator.profiler_rule_configs:
        params = processing.get("RuleParameters", dict())
        rule = config_name = params.get("rule_to_invoke", "")
        if rule == "DetailedProfilerProcessing":
            config_name = processing.get("RuleConfigurationName")
            break
    return os.path.join(
        estimator.latest_training_job.name,
        "rule-output",
        config_name,
        "",
    )
