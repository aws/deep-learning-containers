#!/usr/bin/env python
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

try:
    from builtins import range
except ImportError:
    pass
import tensorflow as tf
import numpy as np
from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import batching
import horovod.tensorflow as hvd
import os
import sys
import time
import argparse
import random
import shutil
import logging
import math
import re
from glob import glob
from operator import itemgetter
from tensorflow.python.util import nest

def rank0log(logger, *args, **kwargs):
    if hvd.rank() == 0:
        if logger:
            logger.info(''.join([str(x) for x in list(args)]))
        else:
            print(*args, **kwargs)


class LayerBuilder(object):
    def __init__(self, activation=None, data_format='channels_last',
                 training=False, use_batch_norm=False, batch_norm_config=None,
                 conv_initializer=None, adv_bn_init=False):
        self.activation = activation
        self.data_format = data_format
        self.training = training
        self.use_batch_norm = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self.conv_initializer = conv_initializer
        self.adv_bn_init = adv_bn_init
        if self.batch_norm_config is None:
            self.batch_norm_config = {
                'decay': 0.9,
                'epsilon': 1e-4,
                'scale': True,
                'zero_debias_moving_mean': False,
            }

    def _conv2d(self, inputs, activation, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=not self.use_batch_norm,
            kernel_initializer=self.conv_initializer,
            activation=None if self.use_batch_norm else activation,
            *args, **kwargs)
        if self.use_batch_norm:
            x = self.batch_norm(x)
            x = activation(x) if activation is not None else x
        return x

    def conv2d_linear_last_bn(self, inputs, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=False,
            kernel_initializer=self.conv_initializer,
            activation=None, *args, **kwargs)
        param_initializers = {
            'moving_mean': tf.zeros_initializer(),
            'moving_variance': tf.ones_initializer(),
            'beta': tf.zeros_initializer(),
        }
        if self.adv_bn_init:
            param_initializers['gamma'] = tf.zeros_initializer()
        else:
            param_initializers['gamma'] = tf.ones_initializer()
        x = self.batch_norm(x, param_initializers=param_initializers)
        return x

    def conv2d_linear(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, None, *args, **kwargs)

    def conv2d(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, self.activation, *args, **kwargs)

    def pad2d(self, inputs, begin, end=None):
        if end is None:
            end = begin
        try:
            _ = begin[1]
        except TypeError:
            begin = [begin, begin]
        try:
            _ = end[1]
        except TypeError:
            end = [end, end]
        if self.data_format == 'channels_last':
            padding = [[0, 0], [begin[0], end[0]], [begin[1], end[1]], [0, 0]]
        else:
            padding = [[0, 0], [0, 0], [begin[0], end[0]], [begin[1], end[1]]]
        return tf.pad(inputs, padding)

    def max_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.max_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def average_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.average_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def dense_linear(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=None)

    def dense(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=self.activation)

    def activate(self, inputs, activation=None):
        activation = activation or self.activation
        return activation(inputs) if activation is not None else inputs

    def batch_norm(self, inputs, **kwargs):
        all_kwargs = dict(self.batch_norm_config)
        all_kwargs.update(kwargs)
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        return tf.contrib.layers.batch_norm(
            inputs, is_training=self.training, data_format=data_format,
            fused=True, **all_kwargs)

    def spatial_average2d(self, inputs):
        shape = inputs.get_shape().as_list()
        if self.data_format == 'channels_last':
            n, h, w, c = shape
        else:
            n, c, h, w = shape
        n = -1 if n is None else n
        x = tf.layers.average_pooling2d(inputs, (h, w), (1, 1),
                                        data_format=self.data_format)
        return tf.reshape(x, [n, c])

    def flatten2d(self, inputs):
        x = inputs
        if self.data_format != 'channel_last':
            # Note: This ensures the output order matches that of NHWC networks
            x = tf.transpose(x, [0, 2, 3, 1])
        input_shape = x.get_shape().as_list()
        num_inputs = 1
        for dim in input_shape[1:]:
            num_inputs *= dim
        return tf.reshape(x, [-1, num_inputs], name='flatten')

    def residual2d(self, inputs, network, units=None, scale=1.0, activate=False):
        outputs = network(inputs)
        c_axis = -1 if self.data_format == 'channels_last' else 1
        h_axis = 1 if self.data_format == 'channels_last' else 2
        w_axis = h_axis + 1
        ishape, oshape = [y.get_shape().as_list() for y in [inputs, outputs]]
        ichans, ochans = ishape[c_axis], oshape[c_axis]
        strides = ((ishape[h_axis] - 1) // oshape[h_axis] + 1,
                   (ishape[w_axis] - 1) // oshape[w_axis] + 1)
        with tf.name_scope('residual'):
            if (ochans != ichans or strides[0] != 1 or strides[1] != 1):
                inputs = self.conv2d_linear(inputs, units, 1, strides, 'SAME')
            x = inputs + scale * outputs
            if activate:
                x = self.activate(x)
        return x


def resnet_bottleneck_v1(builder, inputs, depth, depth_bottleneck, stride,
                         basic=False):
    num_inputs = inputs.get_shape().as_list()[1]
    x = inputs
    with tf.name_scope('resnet_v1'):
        if depth == num_inputs:
            if stride == 1:
                shortcut = x
            else:
                shortcut = builder.max_pooling2d(x, 1, stride)
        else:
            shortcut = builder.conv2d_linear(x, depth, 1, stride, 'SAME')
        if basic:
            x = builder.pad2d(x, 1)
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'VALID')
            x = builder.conv2d_linear(x, depth, 3, 1, 'SAME')
        else:
            x = builder.conv2d(x, depth_bottleneck, 1, 1, 'SAME')
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
            # x = builder.conv2d_linear(x, depth,            1, 1,      'SAME')
            x = builder.conv2d_linear_last_bn(x, depth, 1, 1, 'SAME')
        x = tf.nn.relu(x + shortcut)
        return x


def inference_resnet_v1_impl(builder, inputs, layer_counts, basic=False):
    x = inputs
    x = builder.pad2d(x, 3)
    x = builder.conv2d(x, 64, 7, 2, 'VALID')
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    for i in range(layer_counts[0]):
        x = resnet_bottleneck_v1(builder, x, 256, 64, 1, basic)
    for i in range(layer_counts[1]):
        x = resnet_bottleneck_v1(builder, x, 512, 128, 2 if i == 0 else 1, basic)
    for i in range(layer_counts[2]):
        x = resnet_bottleneck_v1(builder, x, 1024, 256, 2 if i == 0 else 1, basic)
    for i in range(layer_counts[3]):
        x = resnet_bottleneck_v1(builder, x, 2048, 512, 2 if i == 0 else 1, basic)
    return builder.spatial_average2d(x)


def inference_resnet_v1(inputs, nlayer, data_format='channels_last',
                        training=False, conv_initializer=None, adv_bn_init=False):
    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    builder = LayerBuilder(tf.nn.relu, data_format, training, use_batch_norm=True,
                           conv_initializer=conv_initializer, adv_bn_init=adv_bn_init)
    if nlayer == 18:
        return inference_resnet_v1_impl(builder, inputs, [2, 2, 2, 2], basic=True)
    elif nlayer == 34:
        return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3], basic=True)
    elif nlayer == 50:
        return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3])
    elif nlayer == 101:
        return inference_resnet_v1_impl(builder, inputs, [3, 4, 23, 3])
    elif nlayer == 152:
        return inference_resnet_v1_impl(builder, inputs, [3, 8, 36, 3])
    else:
        raise ValueError("Invalid nlayer (%i); must be one of: 18,34,50,101,152" %
                         nlayer)


def get_model_func(model_name):
    if model_name.startswith('resnet'):
        nlayer = int(model_name[len('resnet'):])
        return lambda images, *args, **kwargs: \
            inference_resnet_v1(images, nlayer, *args, **kwargs)
    else:
        raise ValueError("Invalid model type: %s" % model_name)


def deserialize_image_record(record):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, -1),
        'image/class/text': tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label = tf.cast(obj['image/class/label'], tf.int32)
        bbox = tf.stack([obj['image/object/bbox/%s' % x].values
                         for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0, 2, 1])
        text = obj['image/class/text']
        return imgdata, label, bbox, text


def decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')


def crop_and_resize_image(image, original_bbox, height, width, 
                          distort=False, nsummary=10):
    with tf.name_scope('crop_and_resize'):
        # Evaluation is done on a center-crop of this ratio
        eval_crop_ratio = 0.8
        if distort:
            initial_shape = [int(round(height / eval_crop_ratio)),
                             int(round(width / eval_crop_ratio)),
                             3]
            bbox_begin, bbox_size, bbox = \
                tf.image.sample_distorted_bounding_box(
                    initial_shape,
                    bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
                    # tf.zeros(shape=[1,0,4]), # No bounding boxes
                    min_object_covered=0.1,
                    aspect_ratio_range=[3. / 4., 4. / 3.],
                    area_range=[0.08, 1.0],
                    max_attempts=100,
                    seed=11 * hvd.rank(),  # Need to set for deterministic results
                    use_image_if_no_bounding_boxes=True)
            bbox = bbox[0, 0]  # Remove batch, box_idx dims
        else:
            # Central crop
            ratio_y = ratio_x = eval_crop_ratio
            bbox = tf.constant([0.5 * (1 - ratio_y), 0.5 * (1 - ratio_x),
                                0.5 * (1 + ratio_y), 0.5 * (1 + ratio_x)])
        image = tf.image.crop_and_resize(
            image[None, :, :, :], bbox[None, :], [0], [height, width])[0]
        return image


def parse_and_preprocess_image_record(record, counter, height, width,
                                      brightness, contrast, saturation, hue,
                                      distort=False, nsummary=10, increased_aug=False):
    imgdata, label, bbox, text = deserialize_image_record(record)
    label -= 1  # Change to 0-based (don't use background class)
    with tf.name_scope('preprocess_train'):
        try:
            image = decode_jpeg(imgdata, channels=3)
        except:
            image = tf.image.decode_png(imgdata, channels=3)
        image = crop_and_resize_image(image, bbox, height, width, distort)
        if distort:
            image = tf.image.random_flip_left_right(image)
            if increased_aug:
                image = tf.image.random_brightness(image, max_delta=brightness)
                image = distort_image_ops.random_hsv_in_yiq(image, 
                                                            lower_saturation=saturation, 
                                                            upper_saturation=2.0 - saturation, 
                                                            max_delta_hue=hue * math.pi)
                image = tf.image.random_contrast(image, lower=contrast, upper=2.0 - contrast)
                tf.summary.image('distorted_color_image', tf.expand_dims(image, 0))
        image = tf.clip_by_value(image, 0., 255.)
        image = tf.cast(image, tf.uint8)
        return image, label

def make_dataset(filenames, take_count, batch_size, height, width,
                 brightness, contrast, saturation, hue,
                 training=False, num_threads=10, nsummary=10, shard=False, synthetic=False,
                 increased_aug=False):
    if synthetic and training:
        input_shape = [height, width, 3]
        input_element = nest.map_structure(lambda s: tf.constant(0.5, tf.float32, s), tf.TensorShape(input_shape))
        label_element = nest.map_structure(lambda s: tf.constant(1, tf.int32, s), tf.TensorShape([1]))
        element = (input_element, label_element)
        ds = tf.data.Dataset.from_tensors(element).repeat()
    else:
        shuffle_buffer_size = 10000
        num_readers = 1
        if hvd.size() > len(filenames):
            assert (hvd.size() % len(filenames)) == 0
            filenames = filenames * (hvd.size() / len(filenames))

        ds = tf.data.Dataset.from_tensor_slices(filenames)
        if shard:
            # split the dataset into parts for each GPU
            ds = ds.shard(hvd.size(), hvd.rank())

        if not training:
            ds = ds.take(take_count)  # make sure all ranks have the same amount

        if training:
            ds = ds.shuffle(1000, seed=7 * (1 + hvd.rank()))

        ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
        counter = tf.data.Dataset.range(sys.maxsize)
        ds = tf.data.Dataset.zip((ds, counter))
        preproc_func = lambda record, counter_: parse_and_preprocess_image_record(
            record, counter_, height, width, brightness, contrast, saturation, hue,
            distort=training, nsummary=nsummary if training else 0, increased_aug=increased_aug)
        ds = ds.map(preproc_func, num_parallel_calls=num_threads)
        if training:
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buffer_size, seed=5*(1+hvd.rank())))
    ds = ds.batch(batch_size)
    return ds


def stage(tensors):
    """Stages the given tensors in a StagingArea for asynchronous put/get.
    """
    stage_area = data_flow_ops.StagingArea(
        dtypes=[tensor.dtype for tensor in tensors],
        shapes=[tensor.get_shape() for tensor in tensors])
    put_op = stage_area.put(tensors)
    get_tensors = stage_area.get()
    tf.add_to_collection('STAGING_AREA_PUTS', put_op)
    return put_op, get_tensors


class PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i + 1])


class LogSessionRunHook(tf.train.SessionRunHook):
    def __init__(self, global_batch_size, num_records, display_every=10, logger=None):
        self.global_batch_size = global_batch_size
        self.num_records = num_records
        self.display_every = display_every
        self.logger = logger

    def after_create_session(self, session, coord):
        rank0log(self.logger, '  Step Epoch Speed   Loss  FinLoss   LR')
        self.elapsed_secs = 0.
        self.count = 0

    def before_run(self, run_context):
        self.t0 = time.time()
        return tf.train.SessionRunArgs(
            fetches=[tf.train.get_global_step(),
                     'loss:0', 'total_loss:0', 'learning_rate:0'])

    def after_run(self, run_context, run_values):
        self.elapsed_secs += time.time() - self.t0
        self.count += 1
        global_step, loss, total_loss, lr = run_values.results
        if global_step == 1 or global_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.global_batch_size / dt
            epoch = global_step * self.global_batch_size / self.num_records
            self.logger.info('%6i %5.1f %7.1f %6.3f %6.3f %7.5f' %
                             (global_step, epoch, img_per_sec, loss, total_loss, lr))
            self.elapsed_secs = 0.
            self.count = 0


def _fp32_trainvar_getter(getter, name, shape=None, dtype=None,
                          trainable=True, regularizer=None,
                          *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      trainable=trainable,
                      regularizer=regularizer if trainable and 'BatchNorm' not in name and 'batchnorm' not in name and 'batch_norm' not in name and 'Batch_Norm' not in name else None,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        cast_name = name + '/fp16_cast'
        try:
            cast_variable = tf.get_default_graph().get_tensor_by_name(
                cast_name + ':0')
        except KeyError:
            cast_variable = tf.cast(variable, dtype, name=cast_name)
        cast_variable._ref = variable._ref
        variable = cast_variable
    return variable


def fp32_trainable_vars(name='fp32_vars', *args, **kwargs):
    """A varible scope with custom variable getter to convert fp16 trainable
    variables with fp32 storage followed by fp16 cast.
    """
    return tf.variable_scope(
        name, custom_getter=_fp32_trainvar_getter, *args, **kwargs)


class MixedPrecisionOptimizer(tf.train.Optimizer):
    """An optimizer that updates trainable variables in fp32."""

    def __init__(self, optimizer,
                 scale=None,
                 name="MixedPrecisionOptimizer",
                 use_locking=False):
        super(MixedPrecisionOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        if var_list is None:
            var_list = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        replaced_list = var_list

        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)

        gradvar = self._optimizer.compute_gradients(loss, replaced_list, *args, **kwargs)

        final_gradvar = []
        for orig_var, (grad, var) in zip(var_list, gradvar):
            if var is not orig_var:
                grad = tf.cast(grad, orig_var.dtype)
            if self._scale != 1.0:
                grad = tf.scalar_mul(1. / self._scale, grad)
            final_gradvar.append((grad, orig_var))

        return final_gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

class LarcOptimizer(tf.train.Optimizer):
    """ LARC implementation
        -------------------
        Parameters:
          - optimizer:     initial optimizer that you wanna apply
                           example: tf.train.MomentumOptimizer
          - learning_rate: initial learning_rate from initial optimizer
          - clip:          if True apply LARC otherwise LARS
          - epsilon:       default value is weights or grads are 0.
          - name
          - use_locking
    """

    def __init__(self, optimizer, learning_rate, eta, clip=True, epsilon=1.,
                 name="LarcOptimizer", use_locking=False):
        super(LarcOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._learning_rate = learning_rate
        self._eta = float(eta)
        self._clip = clip
        self._epsilon = float(epsilon)

    def compute_gradients(self, *args, **kwargs):
        return self._optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, gradvars, *args, **kwargs):
        v_list = [tf.norm(tensor=v, ord=2) for _, v in gradvars]
        g_list = [tf.norm(tensor=g, ord=2) if g is not None else 0.0
                  for g, _ in gradvars]
        v_norms = tf.stack(v_list)
        g_norms = tf.stack(g_list)
        zeds = tf.zeros_like(v_norms)
        # assign epsilon if weights or grads = 0, to avoid division by zero
        # also prevent biases to get stuck at initialization (0.)
        cond = tf.logical_and(
            tf.not_equal(v_norms, zeds),
            tf.not_equal(g_norms, zeds))
        true_vals = tf.scalar_mul(self._eta, tf.div(v_norms, g_norms))
        # true_vals = tf.scalar_mul(tf.cast(self._eta, tf.float32), tf.div(tf.cast(v_norms, tf.float32), tf.cast(g_norms, tf.float32)))
        false_vals = tf.fill(tf.shape(v_norms), self._epsilon)
        larc_local_lr = tf.where(cond, true_vals, false_vals)
        if self._clip:
            ones = tf.ones_like(v_norms)
            lr = tf.fill(tf.shape(v_norms), self._learning_rate)
            # We need gradients to compute local learning rate,
            # so compute_gradients from initial optimizer have to called
            # for which learning rate is already fixed
            # We then have to scale the gradients instead of the learning rate.
            larc_local_lr = tf.minimum(tf.div(larc_local_lr, lr), ones)
        gradvars = [(tf.multiply(larc_local_lr[i], g), v)
                    if g is not None else (None, v)
                    for i, (g, v) in enumerate(gradvars)]
        return self._optimizer.apply_gradients(gradvars, *args, **kwargs)


def get_with_default(obj, key, default_value):
    return obj[key] if key in obj and obj[key] is not None else default_value


def get_lr(lr, steps, lr_steps, warmup_it, decay_steps, global_step, lr_decay_mode,
           cdr_first_decay_ratio, cdr_t_mul, cdr_m_mul, cdr_alpha, lc_periods, lc_alpha, lc_beta):
    if lr_decay_mode == 'steps':
        learning_rate = tf.train.piecewise_constant(global_step,
                                                    steps, lr_steps)
    elif lr_decay_mode == 'poly' or lr_decay_mode == 'poly_cycle':
        cycle = lr_decay_mode == 'poly_cycle'
        learning_rate = tf.train.polynomial_decay(lr,
                                                  global_step - warmup_it,
                                                  decay_steps=decay_steps - warmup_it,
                                                  end_learning_rate=0.00001,
                                                  power=2,
                                                  cycle=cycle)
    elif lr_decay_mode == 'cosine_decay_restarts':
        learning_rate = tf.train.cosine_decay_restarts(lr, 
                                                       global_step - warmup_it,
                                                       (decay_steps - warmup_it) * cdr_first_decay_ratio,
                                                       t_mul=cdr_t_mul, 
                                                       m_mul=cdr_m_mul,
                                                       alpha=cdr_alpha)
    elif lr_decay_mode == 'cosine':
        learning_rate = tf.train.cosine_decay(lr,
                                              global_step - warmup_it,
                                              decay_steps=decay_steps - warmup_it,
                                              alpha=0.0)
    elif lr_decay_mode == 'linear_cosine':
        learning_rate = tf.train.linear_cosine_decay(lr,
                                                     global_step - warmup_it,
                                                     decay_steps=decay_steps - warmup_it,
                                                     num_periods=lc_periods,#0.47,
                                                     alpha=lc_alpha,#0.0,
                                                     beta=lc_beta)#0.00001)
    else:
        raise ValueError('Invalid type of lr_decay_mode')
    return learning_rate


def warmup_decay(warmup_lr, global_step, warmup_steps, warmup_end_lr):
    from tensorflow.python.ops import math_ops
    p = tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    diff = math_ops.subtract(warmup_end_lr, warmup_lr)
    res = math_ops.add(warmup_lr, math_ops.multiply(diff, p))
    return res


def cnn_model_function(features, labels, mode, params):
    labels = tf.reshape(labels, (-1,))  # Squash unnecessary unary dim
    lr = params['lr']
    lr_steps = params['lr_steps']
    steps = params['steps']
    use_larc = params['use_larc']
    leta = params['leta']
    lr_decay_mode = params['lr_decay_mode']
    decay_steps = params['decay_steps']
    cdr_first_decay_ratio = params['cdr_first_decay_ratio']
    cdr_t_mul = params['cdr_t_mul']
    cdr_m_mul = params['cdr_m_mul']
    cdr_alpha = params['cdr_alpha']
    lc_periods = params['lc_periods']
    lc_alpha = params['lc_alpha']
    lc_beta = params['lc_beta']

    model_name = params['model']
    num_classes = params['n_classes']
    model_dtype = get_with_default(params, 'dtype', tf.float32)
    model_format = get_with_default(params, 'format', 'channels_first')
    device = get_with_default(params, 'device', '/gpu:0')
    model_func = get_model_func(model_name)
    inputs = features  # TODO: Should be using feature columns?
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    momentum = params['mom']
    weight_decay = params['wdecay']
    warmup_lr = params['warmup_lr']
    warmup_it = params['warmup_it']
    loss_scale = params['loss_scale']

    adv_bn_init = params['adv_bn_init']
    conv_init = params['conv_init']

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.device('/cpu:0'):
            preload_op, (inputs, labels) = stage([inputs, labels])

    with tf.device(device):
        if mode == tf.estimator.ModeKeys.TRAIN:
            gpucopy_op, (inputs, labels) = stage([inputs, labels])
        inputs = tf.cast(inputs, model_dtype)
        imagenet_mean = np.array([121, 115, 100], dtype=np.float32)
        imagenet_std = np.array([70, 68, 71], dtype=np.float32)
        inputs = tf.subtract(inputs, imagenet_mean)
        inputs = tf.multiply(inputs, 1. / imagenet_std)
        if model_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])
        with fp32_trainable_vars(
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
            top_layer = model_func(
                inputs, data_format=model_format, training=is_training,
                conv_initializer=conv_init, adv_bn_init=adv_bn_init)
            logits = tf.layers.dense(top_layer, num_classes,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        predicted_classes = tf.argmax(logits, axis=1, output_type=tf.int32)
        logits = tf.cast(logits, tf.float32)
        if mode == tf.estimator.ModeKeys.PREDICT:
            probabilities = tf.softmax(logits)
            predictions = {
                'class_ids': predicted_classes[:, None],
                'probabilities': probabilities,
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)
        loss = tf.identity(loss, name='loss')  # For access by logger (TODO: Better way to access it?)

        if mode == tf.estimator.ModeKeys.EVAL:
            with tf.device(None):  # Allow fallback to CPU if no GPU support for these ops
                accuracy = tf.metrics.accuracy(
                    labels=labels, predictions=predicted_classes)
                top5acc = tf.metrics.mean(
                    tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32))
                newaccuracy = (hvd.allreduce(accuracy[0]), accuracy[1])
                newtop5acc = (hvd.allreduce(top5acc[0]), top5acc[1])
                metrics = {'val-top1acc': newaccuracy, 'val-top5acc': newtop5acc}
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        assert (mode == tf.estimator.ModeKeys.TRAIN)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([loss] + reg_losses, name='total_loss')

        batch_size = tf.shape(inputs)[0]

        global_step = tf.train.get_global_step()

        with tf.device('/cpu:0'):  # Allow fallback to CPU if no GPU support for these ops
            learning_rate = tf.cond(global_step < warmup_it,
                                    lambda: warmup_decay(warmup_lr, global_step, warmup_it,
                                                         lr),
                                    lambda: get_lr(lr, steps, lr_steps, warmup_it, decay_steps, global_step,
                                                   lr_decay_mode, 
                                                   cdr_first_decay_ratio, cdr_t_mul, cdr_m_mul, cdr_alpha, 
                                                   lc_periods, lc_alpha, lc_beta))
            learning_rate = tf.identity(learning_rate, 'learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)

        opt = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_nesterov=True)
        opt = hvd.DistributedOptimizer(opt)
        if use_larc:
            opt = LarcOptimizer(opt, learning_rate, leta, clip=True)
        opt = MixedPrecisionOptimizer(opt, scale=loss_scale)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) or []
        with tf.control_dependencies(update_ops):
            gate_gradients = (tf.train.Optimizer.GATE_NONE)
            train_op = opt.minimize(
                total_loss, global_step=tf.train.get_global_step(),
                gate_gradients=gate_gradients)
        train_op = tf.group(preload_op, gpucopy_op, train_op)  # , update_ops)

        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def get_num_records(filenames):
    def count_records(tf_record_filename):
        count = 0
        for _ in tf.python_io.tf_record_iterator(tf_record_filename):
            count += 1
        return count

    nfile = len(filenames)
    return (count_records(filenames[0]) * (nfile - 1) +
            count_records(filenames[-1]))


def add_bool_argument(cmdline, shortname, longname=None, default=False, help=None):
    if longname is None:
        shortname, longname = None, shortname
    elif default == True:
        raise ValueError("""Boolean arguments that are True by default should not have short names.""")
    name = longname[2:]
    feature_parser = cmdline.add_mutually_exclusive_group(required=False)
    if shortname is not None:
        feature_parser.add_argument(shortname, '--' + name, dest=name, action='store_true', help=help, default=default)
    else:
        feature_parser.add_argument('--' + name, dest=name, action='store_true', help=help, default=default)
    feature_parser.add_argument('--no' + name, dest=name, action='store_false')
    return cmdline


def add_cli_args():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic options
    cmdline.add_argument('-m', '--model', default='resnet50',
                         help="""Name of model to run: resnet[18,34,50,101,152]""")
    cmdline.add_argument('--data_dir',
                         help="""Path to dataset in TFRecord format
                         (aka Example protobufs). Files should be
                         named 'train-*' and 'validation-*'.""")
    add_bool_argument(cmdline, '--synthetic', help="""Whether to use synthetic data for training""")
    cmdline.add_argument('-b', '--batch_size', default=256, type=int,
                         help="""Size of each minibatch per GPU""")
    cmdline.add_argument('--num_batches', type=int,
                         help="""Number of batches to run.
                         Ignored during eval or if num epochs given""")
    cmdline.add_argument('--num_epochs', type=int,
                         help="""Number of epochs to run.
                         Overrides --num_batches. Ignored during eval.""")
    cmdline.add_argument('--log_dir', default='imagenet_resnet',
                         help="""Directory in which to write training
                         summaries and checkpoints. If the log directory already 
                         contains some checkpoints, it tries to resume training
                         from the last saved checkpoint. Pass --clear_log if you
                         want to clear all checkpoints and start a fresh run""")
    add_bool_argument(cmdline, '--clear_log', default=False,
                      help="""Clear the log folder passed so a fresh run can be started""")
    cmdline.add_argument('--log_name', type=str, default='hvd_train.log')
    add_bool_argument(cmdline, '--local_ckpt',
                      help="""Performs local checkpoints (i.e. one per node)""")
    cmdline.add_argument('--display_every', default=50, type=int,
                         help="""How often (in iterations) to print out
                         running information.""")
    add_bool_argument(cmdline, '--eval',
                      help="""Evaluate the top-1 and top-5 accuracy of
                      the latest checkpointed model. If you want to evaluate using multiple GPUs ensure that 
                      all processes have access to all checkpoints. Either if checkpoints 
                      were saved using --local_ckpt or they were saved to a shared directory which all processes
                      can access.""")
    cmdline.add_argument('--eval_interval', type=int,
                         help="""Evaluate accuracy per eval_interval number of epochs""")
    add_bool_argument(cmdline, '--fp16', default=True,
                      help="""Train using float16 (half) precision instead
                      of float32.""")
    cmdline.add_argument('--num_gpus', default=1, type=int,
                         help="""Specify total number of GPUS used to train a checkpointed model during eval.
                                Used only to calculate epoch number to print during evaluation""")

    cmdline.add_argument('--save_checkpoints_steps', type=int, default=1000)
    cmdline.add_argument('--save_summary_steps', type=int, default=0)
    add_bool_argument(cmdline, '--adv_bn_init', default=True,
                      help="""init gamme of the last BN of each ResMod at 0.""")
    add_bool_argument(cmdline, '--adv_conv_init', default=True,
                      help="""init conv with MSRA initializer""")

    cmdline.add_argument('--lr', type=float,
                         help="""Start learning rate""")
    cmdline.add_argument('--mom', default=0.90, type=float,
                         help="""Momentum""")
    cmdline.add_argument('--wdecay', default=0.0001, type=float,
                         help="""Weight decay""")
    cmdline.add_argument('--loss_scale', default=1024., type=float,
                         help="""loss scale""")
    cmdline.add_argument('--warmup_lr', default=0.001, type=float,
                         help="""Warmup starting from this learning rate""")
    cmdline.add_argument('--warmup_epochs', default=0, type=int,
                         help="""Number of epochs in which to warmup to given lr""")
    cmdline.add_argument('--lr_decay_steps', default='30,60,80', type=str,
                         help="""epoch numbers at which lr is decayed by lr_decay_lrs. 
                         Used when lr_decay_mode is steps""")
    cmdline.add_argument('--lr_decay_lrs', default='', type=str,
                         help="""learning rates at specific epochs""")
    cmdline.add_argument('--lr_decay_mode', default='poly',
                         help="""Takes either `steps` (decay by a factor at specified steps) 
                         or `poly`(polynomial_decay with degree 2)""")
    
    add_bool_argument(cmdline, '--use_larc', default=False, 
                        help="""Use Layer wise Adaptive Rate Control which helps convergence at really large batch sizes""")
    cmdline.add_argument('--leta', default=0.013, type=float,
                         help="""The trust coefficient for LARC optimization, LARC Eta""")
    
    cmdline.add_argument('--cdr_first_decay_ratio', default=0.33, type=float,
                         help="""Cosine Decay Restart First Deacy Steps ratio""")
    cmdline.add_argument('--cdr_t_mul', default=2.0, type=float,
                         help="""Cosine Decay Restart t_mul""")
    cmdline.add_argument('--cdr_m_mul', default=0.1, type=float,
                         help="""Cosine Decay Restart m_mul""")
    cmdline.add_argument('--cdr_alpha', default=0.0, type=float,
                         help="""Cosine Decay Restart alpha""")
    cmdline.add_argument('--lc_periods', default=0.47, type=float,
                         help="""Linear Cosine num of periods""")
    cmdline.add_argument('--lc_alpha', default=0.0, type=float,
                         help="""linear Cosine alpha""")
    cmdline.add_argument('--lc_beta', default=0.00001, type=float,
                         help="""Liner Cosine Beta""")

    add_bool_argument(cmdline, '--increased_aug', default=False, 
                         help="""Increase augmentations helpful when training with large number of GPUs such as 128 or 256""")
    cmdline.add_argument('--contrast', default=0.6, type=float,
                         help="""contrast factor""")
    cmdline.add_argument('--saturation', default=0.6, type=float,
                         help="""saturation factor""")
    cmdline.add_argument('--hue', default=0.13, type=float,
                         help="""hue max delta factor, hue delta = hue * math.pi""")
    cmdline.add_argument('--brightness', default=0.3, type=float,
                         help="""Brightness factor""")
    return cmdline


def sort_and_load_ckpts(log_dir):
    ckpts = []
    for f in os.listdir(log_dir):
        m = re.match(r'model.ckpt-([0-9]+).index', f)
        if m is None:
            continue
        fullpath = os.path.join(log_dir, f)
        ckpts.append({'step': int(m.group(1)),
                      'path': os.path.splitext(fullpath)[0],
                      'mtime': os.stat(fullpath).st_mtime,
                      })
    ckpts.sort(key=itemgetter('step'))
    return ckpts


def main():
    gpu_thread_count = 2
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    hvd.init()


    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.force_gpu_compatible = True  # Force pinned memory
    config.intra_op_parallelism_threads = 1  # Avoid pool of Eigen threads
    config.inter_op_parallelism_threads = 5

    # random.seed(5 * (1 + hvd.rank()))
    # np.random.seed(7 * (1 + hvd.rank()))
    # tf.set_random_seed(31 * (1 + hvd.rank()))

    cmdline = add_cli_args()
    FLAGS, unknown_args = cmdline.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")

    FLAGS.data_dir = None if FLAGS.data_dir == "" else FLAGS.data_dir
    FLAGS.log_dir = None if FLAGS.log_dir == "" else FLAGS.log_dir

    if FLAGS.eval:
        FLAGS.log_name = 'eval_' + FLAGS.log_name
    if FLAGS.local_ckpt:
        do_checkpoint = hvd.local_rank() == 0
    else:
        do_checkpoint = hvd.rank() == 0
    if hvd.local_rank() == 0 and FLAGS.clear_log and os.path.isdir(FLAGS.log_dir):
        shutil.rmtree(FLAGS.log_dir)
    barrier = hvd.allreduce(tf.constant(0, dtype=tf.float32))
    tf.Session(config=config).run(barrier)

    if hvd.local_rank() == 0 and not os.path.isdir(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    barrier = hvd.allreduce(tf.constant(0, dtype=tf.float32))
    tf.Session(config=config).run(barrier)
    
    logger = logging.getLogger(FLAGS.log_name)
    logger.setLevel(logging.INFO)  # INFO, ERROR
    # file handler which logs debug messages
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # add formatter to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if not hvd.rank():
        fh = logging.FileHandler(os.path.join(FLAGS.log_dir, FLAGS.log_name))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # add handlers to logger
        logger.addHandler(fh)
    
    height, width = 224, 224
    global_batch_size = FLAGS.batch_size * hvd.size()
    rank0log(logger, 'PY' + str(sys.version) + 'TF' + str(tf.__version__))
    rank0log(logger, "Horovod size: ", hvd.size())

    if FLAGS.data_dir:
        filename_pattern = os.path.join(FLAGS.data_dir, '%s-*')
        train_filenames = sorted(tf.gfile.Glob(filename_pattern % 'train'))
        eval_filenames = sorted(tf.gfile.Glob(filename_pattern % 'validation'))
        num_training_samples = get_num_records(train_filenames)
        rank0log(logger, "Using data from: ", FLAGS.data_dir)
        if not FLAGS.eval:
            rank0log(logger, 'Found ', num_training_samples, ' training samples')
    else:
        if not FLAGS.synthetic:
            raise ValueError('data_dir missing. Please pass --synthetic if you want to run on synthetic data. Else please pass --data_dir')
        train_filenames = eval_filenames = []
        num_training_samples = 1281167
    training_samples_per_rank = num_training_samples // hvd.size()

    if FLAGS.num_epochs:
        nstep = num_training_samples * FLAGS.num_epochs // global_batch_size
    elif FLAGS.num_batches:
        nstep = FLAGS.num_batches
        FLAGS.num_epochs = max(nstep * global_batch_size // num_training_samples, 1)
    else:
        raise ValueError("Either num_epochs or num_batches has to be passed")
    nstep_per_epoch = num_training_samples // global_batch_size
    decay_steps = nstep

    if FLAGS.lr_decay_mode == 'steps':
        steps = [int(x) * nstep_per_epoch for x in FLAGS.lr_decay_steps.split(',')]
        lr_steps = [float(x) for x in FLAGS.lr_decay_lrs.split(',')]
    else:
        steps = []
        lr_steps = []

    if not FLAGS.lr:
        if FLAGS.use_larc:
            FLAGS.lr = 3.7
        else:
            FLAGS.lr = (hvd.size() * FLAGS.batch_size * 0.1) / 256
    if not FLAGS.save_checkpoints_steps:
        # default to save one checkpoint per epoch
        FLAGS.save_checkpoints_steps = nstep_per_epoch
    if not FLAGS.save_summary_steps:
        # default to save one checkpoint per epoch
        FLAGS.save_summary_steps = nstep_per_epoch
    
    if not FLAGS.eval:
        rank0log(logger, 'Using a learning rate of ', FLAGS.lr)
        rank0log(logger, 'Checkpointing every ' + str(FLAGS.save_checkpoints_steps) + ' steps')
        rank0log(logger, 'Saving summary every ' + str(FLAGS.save_summary_steps) + ' steps')

    warmup_it = nstep_per_epoch * FLAGS.warmup_epochs

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_function,
        model_dir=FLAGS.log_dir,
        params={
            'model': FLAGS.model,
            'decay_steps': decay_steps,
            'n_classes': 1000,
            'dtype': tf.float16 if FLAGS.fp16 else tf.float32,
            'format': 'channels_first',
            'device': '/gpu:0',
            'lr': FLAGS.lr,
            'mom': FLAGS.mom,
            'wdecay': FLAGS.wdecay,
            'use_larc': FLAGS.use_larc,
            'leta': FLAGS.leta,
            'steps': steps,
            'lr_steps': lr_steps,
            'lr_decay_mode': FLAGS.lr_decay_mode,
            'warmup_it': warmup_it,
            'warmup_lr': FLAGS.warmup_lr,
            'cdr_first_decay_ratio': FLAGS.cdr_first_decay_ratio,
            'cdr_t_mul': FLAGS.cdr_t_mul,
            'cdr_m_mul': FLAGS.cdr_m_mul,
            'cdr_alpha': FLAGS.cdr_alpha,
            'lc_periods': FLAGS.lc_periods,
            'lc_alpha': FLAGS.lc_alpha,
            'lc_beta': FLAGS.lc_beta,
            'loss_scale': FLAGS.loss_scale,
            'adv_bn_init': FLAGS.adv_bn_init,
            'conv_init': tf.variance_scaling_initializer() if FLAGS.adv_conv_init else None
        },
        config=tf.estimator.RunConfig(
            # tf_random_seed=31 * (1 + hvd.rank()),
            session_config=config,
            save_summary_steps=FLAGS.save_summary_steps if do_checkpoint else None,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps if do_checkpoint else None,
            keep_checkpoint_max=None))

    if not FLAGS.eval:
        num_preproc_threads = 5
        rank0log(logger, "Using preprocessing threads per GPU: ", num_preproc_threads)
        training_hooks = [hvd.BroadcastGlobalVariablesHook(0),
                          PrefillStagingAreasHook()]
        if hvd.rank() == 0:
            training_hooks.append(
                LogSessionRunHook(global_batch_size,
                                  num_training_samples,
                                  FLAGS.display_every, logger))
        try:
            start_time = time.time()
            classifier.train(
                input_fn=lambda: make_dataset(
                    train_filenames,
                    training_samples_per_rank,
                    FLAGS.batch_size, height, width, 
                    FLAGS.brightness, FLAGS.contrast, FLAGS.saturation, FLAGS.hue, 
                    training=True, num_threads=num_preproc_threads, 
                    shard=True, synthetic=FLAGS.synthetic, increased_aug=FLAGS.increased_aug),
                max_steps=nstep,
                hooks=training_hooks)
            rank0log(logger, "Finished in ", time.time() - start_time)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
    elif FLAGS.eval and not FLAGS.synthetic:
        rank0log(logger, "Evaluating")
        rank0log(logger, "Validation dataset size: {}".format(get_num_records(eval_filenames)))
        barrier = hvd.allreduce(tf.constant(0, dtype=tf.float32))
        tf.Session(config=config).run(barrier)
        time.sleep(5)  # a little extra margin...
        if FLAGS.num_gpus == 1:
            rank0log(logger, """If you are evaluating checkpoints of a multi-GPU run on a single GPU,
             ensure you set --num_gpus to the number of GPUs it was trained on.
             This will ensure that the epoch number is accurately displayed in the below logs.""")
        try:
            ckpts = sort_and_load_ckpts(FLAGS.log_dir)
            for i, c in enumerate(ckpts):
                if i < len(ckpts) - 1:
                    if (not FLAGS.eval_interval) or \
                            (i % FLAGS.eval_interval != 0):
                        continue
                eval_result = classifier.evaluate(
                    input_fn=lambda: make_dataset(
                        eval_filenames,
                        get_num_records(eval_filenames), FLAGS.batch_size,
                        height, width, 
                        FLAGS.brightness, FLAGS.contrast, FLAGS.saturation, FLAGS.hue,
                        training=False, shard=True, increased_aug=False),
                    checkpoint_path=c['path'])
                c['epoch'] = c['step'] / (num_training_samples // (FLAGS.batch_size * FLAGS.num_gpus))
                c['top1'] = eval_result['val-top1acc']
                c['top5'] = eval_result['val-top5acc']
                c['loss'] = eval_result['loss']
            rank0log(logger, ' step  epoch  top1    top5     loss   checkpoint_time(UTC)')
            barrier = hvd.allreduce(tf.constant(0, dtype=tf.float32))
            for i, c in enumerate(ckpts):
                tf.Session(config=config).run(barrier)
                if 'top1' not in c:
                    continue
                rank0log(logger,'{:5d}  {:5.1f}  {:5.3f}  {:6.2f}  {:6.2f}  {time}'
                         .format(c['step'],
                                 c['epoch'],
                                 c['top1'] * 100,
                                 c['top5'] * 100,
                                 c['loss'],
                                 time=time.strftime('%Y-%m-%d %H:%M:%S', 
                                    time.localtime(c['mtime']))))
            rank0log(logger, "Finished evaluation")
        except KeyboardInterrupt:
            logger.error("Keyboard interrupt")

if __name__ == '__main__':
    main()
