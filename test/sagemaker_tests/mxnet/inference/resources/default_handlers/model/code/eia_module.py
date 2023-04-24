#  Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import mxnet as mx
import eimx
import logging
import os


def model_fn(model_dir):
    logging.info('Invoking user-defined model_fn')
    # The compiled model artifacts are saved with the prefix 'compiled'
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, 'model'), 0)
    sym = sym.optimize_for('EIA')
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    exe = mod.bind(for_training=False,
               data_shapes=[('data', (1,2))],
               label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    return mod