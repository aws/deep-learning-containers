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
import argparse
import os

import keras
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str)

args = parser.parse_args()


# Loading pre-trained Keras model
model = keras.applications.inception_v3.InceptionV3(weights='imagenet')

# Exports the keras model as TensorFlow Serving Saved Model
with tf.compat.v1.Session() as session:

    init = tf.compat.v1.global_variables_initializer()
    session.run(init)

    tf.compat.v1.saved_model.simple_save(
        session,
        os.path.join(args.model_dir, 'inception-model/1'),
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})
