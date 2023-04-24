import argparse
import gzip
import io
import json
import logging
import os
import struct
from collections import namedtuple
from packaging import version

import mxnet as mx
import numpy as np

def get_context():
    mxnet_version = version.parse(mx.__version__)
    if mxnet_version >= version.parse("1.8"):
        import mx_neuron as neuron
        return mx.cpu()
    else:
        return mx.neuron()

### NOTE: model_fn and transform_fn are used to load the model and serve inference
def model_fn(model_dir):
    logging.info("Invoking user-defined model_fn")
    ctx = get_context()
    Batch = namedtuple("Batch", ["data"])
    dtype = "float32"

    #     print("param {}".format(os.environ.get('MODEL_NAME_CUSTOM')))
    print("ctx {}".format(ctx))
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, "compiled"), 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    for arg in arg_params:
        arg_params[arg] = arg_params[arg].astype(dtype)

    for arg in aux_params:
        aux_params[arg] = aux_params[arg].astype(dtype)

    exe = mod.bind(
        for_training=False, data_shapes=[("data", (1, 28, 28))], label_shapes=mod._label_shapes
    )
    mod.set_params(arg_params, aux_params, allow_missing=True)
    # run warm-up inference on empty data
    data = mx.nd.empty((1, 28, 28), ctx=mx.cpu())
    mod.forward(Batch([data]))
    return mod


def transform_fn(mod, payload, input_content_type, output_content_type):

    logging.info("Invoking user-defined transform_fn")
    logging.info("input_content_type %s", input_content_type)
    Batch = namedtuple("Batch", ["data"])
    ctx = get_context()
    
    print("payload {}".format(payload))
    data = np.array(json.loads(payload))
    print("shape {}".format(data.shape))
    npy_payload = json.loads(payload)
    mx_ndarray = mx.nd.array(npy_payload)
    print("mc_ndarray shape {}".format(mx_ndarray.shape))
    mx_ndarray = mx_ndarray.expand_dims(axis=0)
    inference_payload = mx_ndarray.as_in_context(mx.cpu())

    # prediction/inference
    mod.forward(Batch([inference_payload]))

    # post-processing
    result = mod.get_outputs()[0]
    result = result.asnumpy()
    result = np.squeeze(result)
    result_exp = np.exp(result - np.max(result))
    result = result_exp / np.sum(result_exp)
    output_json = json.dumps(result.tolist())
    output_content_type = "application/json"
    return output_json, output_content_type
