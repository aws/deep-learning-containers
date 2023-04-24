# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
from random import randint
import struct
import sys

import numpy as np
import tensorflow as tf

# Utility functions for generating a recordio encoded file of labeled numpy data
# for testing. Each file contains one or more records. Each record is a TensorFlow
# protobuf Example object. Each object contains an integer label and a numpy array
# encoded as a byte list.

# This file can be used in script mode to generate a single file or be used
# as a module to generate files via build_record_file.

_kmagic = 0xced7230a

padding = {}
for amount in range(4):
    if sys.version_info >= (3,):
        padding[amount] = bytes([0x00 for _ in range(amount)])
    else:
        padding[amount] = bytearray([0x00 for _ in range(amount)])


def write_recordio(f, data, header_flag=0):
    """Writes a single data point as a RecordIO record to the given file."""
    length = len(data)
    f.write(struct.pack('I', _kmagic))
    header = (header_flag << 29) | length
    f.write(struct.pack('I', header))
    pad = (((length + 3) >> 2) << 2) - length
    f.write(data)
    f.write(padding[pad])


def write_recordio_multipart(f, data):
    """Writes a single data point into three multipart records."""
    length = len(data)
    stride = int(length / 3)

    data_start = data[0:stride]
    data_middle = data[stride:2 * stride]
    data_end = data[2 * stride:]

    write_recordio(f, data_start, 1)
    write_recordio(f, data_middle, 2)
    write_recordio(f, data_end, 3)


def string_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))


def label_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_numpy_array(f, feature_name, label, arr, multipart=False):
    feature = {'labels': label_feature(label), feature_name: string_feature(arr)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    if multipart:
        write_recordio_multipart(f, example.SerializeToString())
    else:
        write_recordio(f, example.SerializeToString())


def build_record_file(filename, num_records, dimension, classes=2, data_feature_name='data', multipart=False):
    """Builds a recordio encoded file of TF protobuf Example objects. Each object
    is a labeled numpy array. Each example has two field - a single int64 'label'
    field and a single bytes list field, containing a serialized numpy array.

    Each generated numpy array is a multidimensional normal with
    the specified dimension. The normal distribution is class specific, each class
    has a different mean for the distribution, so it should be possible to learn
    a multiclass classifier on this data. Class means are determnistic - so multiple
    calls to this function with the same number of classes will produce samples drawn
    from the same distribution for each class.

    Args:
        filename - the file to write to
        num_records - how many labeled numpy arrays to generate
        classes - the cardinality of labels
        data_feature_name - the name to give the numpy array in the Example object
        dimension - the size of each numpy array.
    """
    with open(filename, 'wb') as f:
        for i in range(num_records):
            cur_class = i % classes
            loc = int(cur_class - (classes / 2))
            write_numpy_array(f, data_feature_name, cur_class, np.random.normal(loc=loc, size=(dimension,)), multipart)


def build_single_record_file(filename, dimension, classes=2, data_feature_name='data'):
    cur_class = randint(0, classes - 1)
    loc = int(cur_class - (classes / 2))

    arr = np.random.normal(loc=loc, size=(dimension,))
    feature = {'labels': label_feature(cur_class), data_feature_name: string_feature(arr)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    with open(filename, 'wb') as f:
        f.write(example.SerializeToString())


def validate_record_file(filename, dimension):
    data = open(filename, 'rb').read()
    magic_number, length = struct.unpack('II', data[0:8])
    encoded = data[8:8 + length]

    features = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(encoded, features)
    array = tf.io.decode_raw(parsed['data'], tf.float64)

    assert array.shape[0] == dimension


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic multi-class training data")
    parser.add_argument('--dimension', default=65536, type=int)
    parser.add_argument('--classes', default=2, type=int)
    parser.add_argument('--num-records', default=4, type=int)
    parser.add_argument('--data-feature-name', default='data')
    parser.add_argument('filename', type=str)
    args = parser.parse_args()
    build_record_file(args.filename, args.num_records, args.dimension, args.classes, args.data_feature_name)
    validate_record_file(args.filename, args.dimension)
