# Copyright 2019-2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""SageMaker hosting with the native Torch NeuronX (2.13+) Hugging Face inference DLC.

Models run on the `neuron` device without precompiled artifacts:
- zero-code deployments (HF_MODEL_ID + HF_TASK) run in eager mode;
- a custom handler runs in eager mode and with `torch.compile(backend="neuron")`.
"""
from __future__ import absolute_import

import os
import shutil
import tarfile
from pathlib import Path

import boto3
import pytest
import sagemaker
from packaging.version import Version
from sagemaker.huggingface import HuggingFaceModel

from test.test_utils import get_framework_and_version_from_tag, get_sha_of_an_image_from_ecr

from ...integration import dump_logs_from_cloudwatch
from ...integration.sagemaker.timeout import timeout_and_delete_endpoint
from ..... import invoke_sm_endpoint_helper_function

DEFAULT_INSTANCE_TYPE = "ml.trn2.48xlarge"
INFERENCE_AMI_VERSION = "al2-ami-sagemaker-inference-neuron-2"
# Shared with the EC2 test, which serves the same handler through the image entrypoint.
HANDLER_PATH = (
    Path(__file__).resolve().parents[5]
    / "dlc_tests/container_tests/bin/huggingface/neuron_inference_handler.py"
)

ZERO_CODE_CASES = {
    "feature-extraction": (
        "sentence-transformers/all-MiniLM-L6-v2",
        {"inputs": "Suffs is considered to be the best musical after Hamilton."},
    ),
    "text-generation": (
        "Qwen/Qwen3-0.6B",
        {
            "inputs": "The capital of France is",
            "parameters": {"max_new_tokens": 8, "do_sample": False},
        },
    ),
}


def _skip_unless_native_neuron(ecr_image):
    framework, version = get_framework_and_version_from_tag(ecr_image)
    if "pytorch" not in framework:
        pytest.skip(f"Skipping test for non-pytorch image - {ecr_image}")
    if Version(version) < Version("2.13"):
        pytest.skip("Native Torch NeuronX inference requires a PyTorch 2.13+ image")


@pytest.mark.model("all-MiniLM-L6-v2-qwen3")
@pytest.mark.processor("neuronx")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
@pytest.mark.parametrize("task", sorted(ZERO_CODE_CASES))
def test_neuronx_native_zero_code(
    task, instance_type, framework_version, ecr_image, sagemaker_regions, py_version
):
    _skip_unless_native_neuron(ecr_image)
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_zero_code,
        framework_version=framework_version,
        instance_type=instance_type or DEFAULT_INSTANCE_TYPE,
        py_version=py_version,
        task=task,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


@pytest.mark.model("distilbert-sst-2")
@pytest.mark.processor("neuronx")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_neuronx_native_custom_handler(
    mode, instance_type, framework_version, ecr_image, sagemaker_regions, py_version, tmp_path
):
    _skip_unless_native_neuron(ecr_image)
    invoke_sm_endpoint_helper_function(
        ecr_image=ecr_image,
        sagemaker_regions=sagemaker_regions,
        test_function=_test_custom_handler,
        framework_version=framework_version,
        instance_type=instance_type or DEFAULT_INSTANCE_TYPE,
        py_version=py_version,
        mode=mode,
        local_dir=tmp_path,
        dump_logs_from_cloudwatch=dump_logs_from_cloudwatch,
    )


def _get_endpoint_name(custom_prefix, region_name, image_uri):
    """
    Creates an endpoint name that has the first 10 chars of the image sha and the
    CODEBUILD_RESOLVED_SOURCE_VERSION to allow tracking of SM Endpoint Logs.
    """
    endpoint_name_prefix = custom_prefix
    try:
        image_sha = get_sha_of_an_image_from_ecr(
            ecr_client=boto3.Session(region_name=region_name).client("ecr"), image_uri=image_uri
        )
        endpoint_name_prefix = f"{endpoint_name_prefix}-{image_sha.split(':')[-1][:10]}"
    except Exception:
        pass
    resolved_src_version = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "temp")
    return sagemaker.utils.unique_name_from_base(f"{endpoint_name_prefix}-{resolved_src_version}")


def _deploy_and_predict(hf_model, sagemaker_session, image_uri, instance_type, prefix, data):
    endpoint_name = _get_endpoint_name(prefix, sagemaker_session.boto_region_name, image_uri)
    with timeout_and_delete_endpoint(endpoint_name, sagemaker_session, minutes=60):
        predictor = hf_model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            inference_ami_version=INFERENCE_AMI_VERSION,
            container_startup_health_check_timeout=1800,
        )
        return predictor.predict(data)


def _test_zero_code(sagemaker_session, image_uri, instance_type, task, **kwargs):
    model_id, data = ZERO_CODE_CASES[task]
    hf_model = HuggingFaceModel(
        env={"HF_MODEL_ID": model_id, "HF_TASK": task},
        role="SageMakerRole",
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
    )
    output = _deploy_and_predict(
        hf_model, sagemaker_session, image_uri, instance_type, f"sm-hf-neuronx-{task}", data
    )
    if task == "feature-extraction":
        embeddings = output["embeddings"]
        assert len(embeddings) == 1 and len(embeddings[0]) == 384, output
    else:
        assert output["generated_text"].strip(), output


def _test_custom_handler(sagemaker_session, image_uri, instance_type, mode, local_dir, **kwargs):
    # model.tar.gz holds only code/handler.py; the handler pulls the weights from the Hub.
    code_dir = Path(local_dir) / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy(HANDLER_PATH, code_dir / "handler.py")
    model_archive = Path(local_dir) / "model.tar.gz"
    with tarfile.open(model_archive, "w:gz") as archive:
        archive.add(code_dir, arcname="code")
    model_data = sagemaker_session.upload_data(
        path=str(model_archive),
        key_prefix=f"sagemaker-huggingface-neuronx-native-{mode}/models",
    )

    hf_model = HuggingFaceModel(
        model_data=model_data,
        env={"HF_TRUST_REMOTE_CODE": "true", "NEURON_INFERENCE_MODE": mode},
        role="SageMakerRole",
        image_uri=image_uri,
        sagemaker_session=sagemaker_session,
    )
    output = _deploy_and_predict(
        hf_model,
        sagemaker_session,
        image_uri,
        instance_type,
        f"sm-hf-neuronx-handler-{mode}",
        {"inputs": "Hamilton is considered to be the best musical of human history."},
    )
    assert output["device"] == "neuron", output
    assert output["mode"] == mode, output
    assert output["label"] == "POSITIVE", output
