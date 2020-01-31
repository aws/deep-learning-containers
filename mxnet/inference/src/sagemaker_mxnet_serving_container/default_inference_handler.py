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

import mxnet as mx
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

from sagemaker_mxnet_serving_container.utils import get_default_context, read_data_shapes

PREFERRED_BATCH_SIZE_PARAM = 'SAGEMAKER_DEFAULT_MODEL_FIRST_DIMENSION_SIZE'
INFERENCE_ACCELERATOR_PRESENT_ENV = 'SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT'

DEFAULT_MODEL_NAME = 'model'
DEFAULT_MODEL_FILENAMES = {
    'symbol': 'model-symbol.json',
    'params': 'model-0000.params',
    'shapes': 'model-shapes.json',
}


class DefaultMXNetInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON, content_types.NPY)

    def default_model_fn(self, model_dir, preferred_batch_size=1):
        """Function responsible for loading the model. This implementation is designed to work with
        the default save function provided for MXNet training.

        Args:
            model_dir (str): The directory where model files are stored
            preferred_batch_size (int): preferred batch size of the model's data shape.
                Defaults to 1.

        Returns:
            mxnet.mod.Module: the loaded model.

        """
        for f in DEFAULT_MODEL_FILENAMES.values():
            path = os.path.join(model_dir, f)
            if not os.path.exists(path):
                raise ValueError('Failed to load model with default model_fn: missing file {}.'
                                 'Expected files: {}'.format(f, [file_name for _, file_name
                                                                 in DEFAULT_MODEL_FILENAMES.items()]))

        shapes_file = os.path.join(model_dir, DEFAULT_MODEL_FILENAMES['shapes'])
        preferred_batch_size = preferred_batch_size or os.environ.get(PREFERRED_BATCH_SIZE_PARAM)
        data_names, data_shapes = read_data_shapes(shapes_file, preferred_batch_size)

        sym, args, aux = mx.model.load_checkpoint(os.path.join(model_dir, DEFAULT_MODEL_NAME), 0)

        ctx = mx.eia() if os.environ.get(INFERENCE_ACCELERATOR_PRESENT_ENV) == 'true' else get_default_context()

        mod = mx.mod.Module(symbol=sym, context=ctx, data_names=data_names, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes)
        mod.set_params(args, aux, allow_missing=True)

        return mod

    def default_input_fn(self, input_data, content_type):
        """Take request data and deserialize it into an MXNet NDArray for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:

            - The request's content type, for example "application/json"
            - The request data

        The ``input_fn`` is responsible for preprocessing request data before prediction.

        Args:
            input_data (obj): the request data
            content_type (str): the request's content type

        Returns:
            mxnet.nd.array: an MXNet NDArray

        Raises:
            sagemaker_inference.errors.UnsupportedFormatError: if an unsupported content type is used.

        """
        if content_type in self.VALID_CONTENT_TYPES:
            np_array = decoder.decode(input_data, content_type)
            return mx.nd.array(np_array).as_in_context(get_default_context())
        else:
            raise errors.UnsupportedFormatError(content_type)

    def default_output_fn(self, prediction, accept):
        """Serialize the prediction into a response.

        Args:
            prediction (mxnet.nd.array): an MXNet NDArray that is the result of a prediction
            accept (str): the accept content type expected by the client

        Returns:
            obj: prediction data.

        Raises:
            sagemaker_inference.errors.UnsupportedFormatError: if an unsupported content type is used.

        """
        if accept in self.VALID_CONTENT_TYPES:
            return encoder.encode(prediction.asnumpy().tolist(), accept)
        else:
            raise errors.UnsupportedFormatError(accept)


class DefaultModuleInferenceHandler(DefaultMXNetInferenceHandler):
    VALID_CONTENT_TYPES = (content_types.JSON, content_types.CSV, content_types.NPY)

    def default_input_fn(self, input_data, content_type, model=None):
        """Take request data and deserialize it into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:

            - The request's content type, for example "application/json"
            - The request data

        The ``input_fn`` is responsible for preprocessing request data before prediction.

        Args:
            input_data (obj): the request data
            content_type (str): the request's content type
            model (obj): an MXNet model

        Returns:
            mxnet.io.NDArrayIter: data ready for prediction.

        Raises:
            sagemaker_inference.errors.UnsupportedFormatError: if an unsupported content type is used.

        """
        if content_type not in self.VALID_CONTENT_TYPES:
            raise errors.UnsupportedFormatError(content_type)

        np_array = decoder.decode(input_data, content_type)
        ndarray = mx.nd.array(np_array).as_in_context(get_default_context())

        # We require model to only have one input
        [data_shape] = model.data_shapes

        # Reshape flattened CSV as specified by the model
        if content_type == content_types.CSV:
            _, target_shape = data_shape
            ndarray = ndarray.reshape(target_shape)

        # Batch size is first dimension of model input
        model_batch_size = data_shape[1][0]
        pad_rows = max(0, model_batch_size - ndarray.shape[0])

        # If ndarray has fewer rows than model_batch_size, then pad it with zeros.
        if pad_rows:
            padding_shape = tuple([pad_rows] + list(ndarray.shape[1:]))
            padding = mx.ndarray.zeros(shape=padding_shape)
            ndarray = mx.ndarray.concat(ndarray, padding, dim=0)

        model_input = mx.io.NDArrayIter(ndarray, batch_size=model_batch_size, last_batch_handle='pad')

        if pad_rows:
            # Update the getpad method on the model_input data iterator to return the amount of
            # padding. MXNet will ignore the last getpad() rows during Module predict.
            def _getpad():
                return pad_rows

            model_input.getpad = _getpad

        return model_input

    def default_predict_fn(self, data, model):
        """Use the model to create a prediction for the data.

        Args:
            data (mxnet.io.NDArrayIter): input data for prediction
            model (mxnet.module.BaseModule): an MXNet Module

        Returns:
            list: the prediction result. This will be either a list of ``mxnet.nd.array`` or
                a list of lists of ``mxnet.nd.array``

        """
        return model.predict(data)


class DefaultGluonBlockInferenceHandler(DefaultMXNetInferenceHandler):
    def default_predict_fn(self, data, block):
        """Use the model to create a prediction for the data.

        Args:
            data (mxnet.nd.array): input data for prediction (deserialized by ``input_fn``)
            block (mxnet.gluon.block.Block): a Gluon neural network

        Returns:
            mxnet.nd.array: the prediction result

        """
        return block(data)
