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
'''Adapted from the Keras-MXNet example found at
https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/mnist_cnn.py
'''
from __future__ import absolute_import, print_function

import argparse
import json
import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np


def main(batch_size, epochs, num_classes, training_channel, model_dir):
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    dataset = np.load(os.path.join(training_channel, 'mnist.npz'))
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        raise ValueError('Unexpected image data format (expected "channels_first"): {}'
                         .format(K.image_data_format()))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('Saving model in MXNet format')
    model_prefix = os.path.join(model_dir, 'model')
    data_name, data_shapes = keras.models.save_mxnet_model(model=model, prefix=model_prefix,
                                                           epoch=0)

    signature = [{'name': data_name[0], 'shape': [dim for dim in data_desc.shape]}
                 for data_desc in data_shapes]
    with open(os.path.join(model_dir, 'model-shapes.json'), 'w') as f:
        json.dump(signature, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_classes', type=float, default=12)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    main(args.batch_size, args.epochs, args.num_classes, args.train, args.model_dir)
