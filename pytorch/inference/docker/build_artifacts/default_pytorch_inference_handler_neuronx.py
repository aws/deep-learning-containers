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

import torch
import torch_neuronx
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
    utils,
)

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
DEFAULT_MODEL_FILENAME = "model.pt"


class ModelLoadError(Exception):
    pass


class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)

    @staticmethod
    def _is_model_file(filename):
        is_model_file = False
        if os.path.isfile(filename):
            _, ext = os.path.splitext(filename)
            is_model_file = ext in [".pt", ".pth"]
        return is_model_file

    def default_model_fn(self, model_dir):
        """Loads a model. For PyTorch, a default function to load a model only if Elastic Inference is used.
        In other cases, users should provide customized model_fn() in script.

        Args:
            model_dir: a directory where model is saved.

        Returns: A PyTorch model.
        """
        if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
            model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
            if not os.path.exists(model_path):
                raise FileNotFoundError("Failed to load model with default model_fn: missing file {}."
                                        .format(DEFAULT_MODEL_FILENAME))
            # Client-framework is CPU only. But model will run in Elastic Inference server with CUDA.
            try:
                return torch.jit.load(model_path, map_location=torch.device('cpu'))
            except RuntimeError as e:
                raise ModelLoadError(
                    "Failed to load {}. Please ensure model is saved using torchscript.".format(model_path)
                ) from e
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
            if not os.path.exists(model_path):
                model_files = [file for file in os.listdir(model_dir) if self._is_model_file(file)]
                if len(model_files) != 1:
                    raise ValueError(
                        "Exactly one .pth or .pt file is required for PyTorch models: {}".format(model_files)
                    )
                model_path = os.path.join(model_dir, model_files[0])
            print(f"DBG: Loading model {model_path}")
            try:
                model = torch.jit.load(model_path, map_location=device)
            except RuntimeError as e:
                raise ModelLoadError(
                    "Failed to load {}. Please ensure model is saved using torchscript.".format(model_path)
                ) from e
            model = model.to(device)
            print("DBG: Loading model done")
            return model

    def default_input_fn(self, input_data, content_type):
        """A default input_fn that can handle JSON, CSV and NPZ formats.

        Args:
            input_data: the request payload serialized in the content_type format
            content_type: the request content_type

        Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor,
            depending if cuda is available.
        """
        print("DBG: Input fn")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            np_array = decoder.decode(input_data, content_type)
            tensor = torch.FloatTensor(
                np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)
        except Exception as e:
            print(f"DBG: Error in input fn {e}")
            raise e
        return tensor.to(device)

    def default_predict_fn(self, data, model):
        """A default predict_fn for PyTorch. Calls a model on data deserialized in input_fn.
        Runs prediction on GPU if cuda is available.

        Args:
            data: input data (torch.Tensor) for prediction deserialized by input_fn
            model: PyTorch model loaded in memory by model_fn

        Returns: a prediction
        """
        print("DBG: Executing prediction")
        try:
            with torch.no_grad():
                if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
                    device = torch.device("cpu")
                    model = model.to(device)
                    input_data = data.to(device)
                    model.eval()
                    with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                        output = model(input_data)
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = model.to(device)
                    input_data = data.to(device)
                    model.eval()
                    output = model(input_data)
        except Exception as e:
            print(f"DBG: Error in predict fn {e}")
            raise e
        print("DBG: Prediction done")
        return output

    def default_output_fn(self, prediction, accept):
        """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

        Args:
            prediction: a prediction result from predict_fn
            accept: type which the output data needs to be serialized

        Returns: output data serialized
        """
        print("DBG: Output fn")
        try:
            if type(prediction) == torch.Tensor:
                prediction = prediction.detach().cpu().numpy().tolist()

            for content_type in utils.parse_accept(accept):
                if content_type in encoder.SUPPORTED_CONTENT_TYPES:
                    encoded_prediction = encoder.encode(prediction, content_type)
                    if content_type == content_types.CSV:
                        encoded_prediction = encoded_prediction.encode("utf-8")
                    print("DBG: Output fn done")
                    return encoded_prediction
        except Exception as e:
            print(f"DBG: Error in predict fn {e}")
            raise e
        raise errors.UnsupportedFormatError(accept)
