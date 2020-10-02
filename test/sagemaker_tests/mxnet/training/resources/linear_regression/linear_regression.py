#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import absolute_import

import argparse
import logging
import json
import os

import mxnet as mx
import numpy as np

from sagemaker_mxnet_container.training_utils import save


def train(num_cpus, num_gpus, channel_input_dirs):
    """
    ensure mxnet is fully functional by training simple model
    see http://mxnet.incubator.apache.org/tutorials/python/linear-regression.html
    """
    print('entered train fn')
    try:
        ctx = _get_context(num_cpus, num_gpus)

        # load training data
        train_data = np.loadtxt(os.path.join(channel_input_dirs['training'], 'train_data.txt'))
        train_label = np.loadtxt(os.path.join(channel_input_dirs['training'], 'train_label.txt'))
        eval_data = np.loadtxt(os.path.join(channel_input_dirs['evaluation'], 'eval_data.txt'))
        eval_label = np.loadtxt(os.path.join(channel_input_dirs['evaluation'], 'eval_label.txt'))

        batch_size = 1
        if isinstance(ctx, list):
            batch_size = len(ctx)

        train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True,
                                       label_name='lin_reg_label')
        eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

        x = mx.sym.Variable('data')
        y = mx.symbol.Variable('lin_reg_label')
        fully_connected_layer = mx.sym.FullyConnected(data=x, name='fc1', num_hidden=1)
        lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=y, name="lro")

        epochs = 1
        model = mx.mod.Module(
            symbol=lro,
            context=ctx,
            data_names=['data'],
            label_names=['lin_reg_label']  # network structure
        )

        model.fit(train_iter, eval_iter,
                  optimizer_params={'learning_rate': 0.005, 'momentum': 0.9},
                  num_epoch=epochs,
                  eval_metric='mse',
                  batch_end_callback=mx.callback.Speedometer(batch_size, 2))

        return model

    except Exception as e:
        print(e)
        logging.exception(e)
        raise e


def _get_context(cpus, gpus):
    if gpus > 0:
        ctx = [mx.gpu(x) for x in range(gpus)]
    else:
        ctx = mx.cpu()

    logging.info("mxnet context: %s" % str(ctx))
    return ctx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--input-channels', type=str, default=json.loads(os.environ['SM_TRAINING_ENV'])['channel_input_dirs'])

    args = parser.parse_args()

    num_cpus = int(os.environ['SM_NUM_CPUS'])
    num_gpus = int(os.environ['SM_NUM_GPUS'])

    model = train(num_cpus, num_gpus, args.input_channels)
    save(args.model_dir, model)
