# Copyright 2019-2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from src.benchmark_metrics import HF_PERFORMANCE_TEST_LATENCY_THRESHOLD

import pytest
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import time
import numpy as np
import signal
import logging
from test import test_utils

warmup_iteration = 20
BERT_BASE_UNCASED_PT_MODEL = "s3://aws-dlc-sample-models/huggingface_models/pytorch/bert-base-uncased-pytorch.tar.gz"
BERT_BASE_UNCASED_TF_MODEL = "s3://aws-dlc-sample-models/huggingface_models/tensorflow/bert-base-uncased-tensorflow.tar.gz"
ROBERTA_BASE_PT_MODEL = "s3://aws-dlc-sample-models/huggingface_models/pytorch/roberta-base-pytorch.tar.gz"
ROBERTA_BASE_TF_MODEL = "s3://aws-dlc-sample-models/huggingface_models/tensorflow/roberta-base-tensorflow.tar.gz"


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


@pytest.mark.model("bert-base-uncased-pt")
def test_sm_pytorch_trained_model_bert_base_uncased_cpu(huggingface_pytorch_inference, huggingface_only):
    framework = "pytorch"
    device = test_utils.get_processor_from_image_uri(huggingface_pytorch_inference)
    instance_type = "ml.p2.xlarge" if device == "gpu" else "ml.m5.xlarge"
    model = BERT_BASE_UNCASED_PT_MODEL
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a [MASK] model."}
    latencies = _test_sm_trained_model(
        instance_type, huggingface_pytorch_inference, model, task, input_data, framework, device
    )
    assert np.average(latencies) <= HF_PERFORMANCE_TEST_LATENCY_THRESHOLD[framework][device]["bert-base-uncased-pt"]


@pytest.mark.model("bert-base-uncased-tf")
def test_sm_tensorflow_trained_model_bert_base_uncased_cpu(huggingface_tensorflow_inference, huggingface_only):
    framework = "tensorflow"
    device = test_utils.get_processor_from_image_uri(huggingface_tensorflow_inference)
    instance_type = "ml.p2.xlarge" if device == "gpu" else "ml.m5.xlarge"
    model = BERT_BASE_UNCASED_TF_MODEL
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a [MASK] model."}
    latencies = _test_sm_trained_model(
        instance_type, huggingface_tensorflow_inference, model, task, input_data, framework, device
    )
    assert np.average(latencies) <= HF_PERFORMANCE_TEST_LATENCY_THRESHOLD[framework][device]["bert-base-uncased-tf"]


@pytest.mark.model("roberta-base-pt")
def test_sm_pytorch_trained_model_roberta_base_cpu(huggingface_pytorch_inference, huggingface_only):
    framework = "pytorch"
    device = test_utils.get_processor_from_image_uri(huggingface_pytorch_inference)
    instance_type = "ml.p2.xlarge" if device == "gpu" else "ml.m5.xlarge"
    model = ROBERTA_BASE_PT_MODEL
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a <mask> model."}
    latencies = _test_sm_trained_model(
        instance_type, huggingface_pytorch_inference, model, task, input_data, framework, device
    )
    assert np.average(latencies) <= HF_PERFORMANCE_TEST_LATENCY_THRESHOLD[framework][device]["roberta-base-pt"]


@pytest.mark.model("roberta-base-tf")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_tensorflow_trained_model_roberta_base_cpu(huggingface_tensorflow_inference, huggingface_only):
    framework = "tensorflow"
    device = test_utils.get_processor_from_image_uri(huggingface_tensorflow_inference)
    instance_type = "ml.p2.xlarge" if device == "gpu" else "ml.m5.xlarge"
    model = ROBERTA_BASE_TF_MODEL
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a <mask> model."}
    latencies = _test_sm_trained_model(
        instance_type, huggingface_tensorflow_inference, model, task, input_data, framework, device
    )
    assert np.average(latencies) <= HF_PERFORMANCE_TEST_LATENCY_THRESHOLD[framework][device]["roberta-base-tf"]


def _test_sm_trained_model(instance_type, ecr_image, model, task, input_data, framework, device, accelerator_type=None):
    hf_model = Model(
        model_data=model,
        role="SageMakerRole",
        image_uri=ecr_image,
        predictor_cls=Predictor,
        env={'HF_TASK': task}
    )
    predictor = hf_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type
    )
    data = input_data
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1800)
    latencies = []
    try:
        for i in range(500 + warmup_iteration):
            start = time.time()
            predictor.predict(data)
            if i > warmup_iteration:
                latencies.append((time.time() - start) * 1000)
    except TimeoutException:
        test_utils.LOGGER.error("Timeout {}, {}, {}.".format(model, framework, device))
        raise TimeoutException("Timeout {}, {}, {}.".format(model, framework, device))
    finally:
        predictor.delete_endpoint()
    return latencies
