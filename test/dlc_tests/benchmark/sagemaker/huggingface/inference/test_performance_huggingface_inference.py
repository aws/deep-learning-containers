# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import os

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

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

@pytest.mark.model("bert-base-uncased-pt")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu():
    instance_type = "ml.m5.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/pytorch/bert-base-uncased-pytorch.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a [MASK] model."}
    framework = "pytorch"
    device = "cpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("bert-base-uncased-pt")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu():
    instance_type = "ml.p2.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/pytorch/bert-base-uncased-pytorch.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a [MASK] model."}
    framework = "pytorch"
    device = "gpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("bert-base-uncased-tf")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu():
    instance_type = "ml.m5.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/tensorflow/bert-base-uncased-tensorflow.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a [MASK] model."}
    framework = "tensorflow"
    device = "cpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("bert-base-uncased-tf")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu():
    instance_type = "ml.p2.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/tensorflow/bert-base-uncased-tensorflow.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a [MASK] model."}
    framework = "tensorflow"
    device = "gpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("roberta-base-pt")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu():
    instance_type = "ml.m5.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/pytorch/roberta-base-pytorch.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a <mask> model."}
    framework = "pytorch"
    device = "cpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("roberta-base-pt")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu():
    instance_type = "ml.p2.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/pytorch/roberta-base-pytorch.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a <mask> model."}
    framework = "pytorch"
    device = "gpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("roberta-base-tf")
@pytest.mark.processor("cpu")
@pytest.mark.cpu_test
def test_sm_trained_model_cpu():
    instance_type = "ml.m5.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/tensorflow/roberta-base-tensorflow.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a <mask> model."}
    framework = "tensorflow"
    device = "cpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

@pytest.mark.model("roberta-base-tf")
@pytest.mark.processor("gpu")
@pytest.mark.gpu_test
def test_sm_trained_model_gpu():
    instance_type = "ml.p2.xlarge"
    model = "s3://aws-dlc-sample-models/huggingface_models/tensorflow/roberta-base-tensorflow.tar.gz"
    task = "fill-mask"
    input_data = {"inputs": "Hello I'm a <mask> model."}
    framework = "tensorflow"
    device = "gpu"
    latencies = _test_sm_trained_model(instance_type, model, task, input_data, framework, device)
    assert np.average(latencies) <= 350

def _test_sm_trained_model(instance_type, model, task, input_data, framework, device, accelerator_type=None):
    ecr_image = ""
    if framework == "pytorch" and device == "cpu":
        ecr_image = "669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-huggingface-pytorch-inference:1.7.1-transformers4.6.1-cpu-py36-ubuntu18.04-beta-tested-2021-06-18-16-02-51"
    elif framework == "pytorch" and device == "gpu":
        ecr_image = "669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-huggingface-pytorch-inference:1.7.1-transformers4.6.1-gpu-py36-cu110-ubuntu18.04-beta-tested-2021-06-18-16-02-51"
    elif framework == "tensorflow" and device == "cpu":
        ecr_image = "669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-huggingface-tensorflow-inference:2.4.1-transformers4.6.1-cpu-py37-ubuntu18.04-beta-tested-2021-06-18-16-27-00"
    else :
        ecr_image = "669063966089.dkr.ecr.us-west-2.amazonaws.com/beta-huggingface-tensorflow-inference:2.4.1-transformers4.6.1-gpu-py37-cu110-ubuntu18.04-beta-tested-2021-06-18-16-27-00"

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
    LOGGER = logging.getLogger('timeout') 
    try:
        for i in range(520):
            start = time.time()
            predictor.predict(data)
            if i > 20:  # Warmup 20 iterations
                latencies.append((time.time() - start) * 1000)
    except TimeoutException:
        LOGGER.info("Timeout {}, {}, {}.".format(model, framework, device))
    finally:
        predictor.delete_endpoint()
    return latencies