#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: inference
Created: 2020-08-19

Description:

    inference functions for pytorch serving container

Usage:

    >>> import inference
    >>> model = inference.model_fn()

"""
import datetime
import json
import os
import sys
import tempfile

import boto3
import cv2

HERE = os.path.dirname(os.path.realpath(__file__))
DSFD_DIR = os.path.join(HERE, 'DSFD-Pytorch-Inference')
sys.path.insert(0, DSFD_DIR)
import face_detection

from sagemaker_inference import content_types, decoder

PLAIN_TEXT = 'text/plain'

RETINA_NAME = 'RetinaNetResNet50'
MODEL_NAME = os.getenv('MODEL_NAME', RETINA_NAME)
if MODEL_NAME not in face_detection.build.available_detectors:
    raise ValueError(f"MODEL_NAME = {MODEL_NAME} is invalid. please choose "
                     f"from one of {face_detection.build.available_detectors}")

WITH_LANDMARKS = os.getenv('WITH_LANDMARKS') == 'TRUE'

# todo: allow input json records to parameterize these; add a block to the start
#  or predict_fn where the model.confidence_threshold or model.nms_iou_threshold
#  is set to 0.5 or a provided value. might not work because we assume that the
#  data we pass around is just a pyspark tensor...
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
NMS_IOU_THRESHOLD = float(os.getenv('NMS_IOU_THRESHOLD', 0.3))


def log_storyblocks_artifact_info():
    """if a face detection repo sha or archive timestamp file exist, print
    their contents"""
    print('looking for archive artifact metadata')
    here = os.path.dirname(os.path.realpath(__file__))

    for fname in ['model_archive_timestamp', 'yaefd_current_sha']:
        fname_full = os.path.join(here, fname)
        if os.path.isfile(fname_full):
            with open(fname_full, 'r') as fp:
                print(f"{fname}: {fp.read().strip()}")


log_storyblocks_artifact_info()


def timer(func):
    def wrapped_func(*args, **kwargs):
        t0 = datetime.datetime.now()
        print(f'entering {func.__name__} at {t0}')
        x = func(*args, **kwargs)
        t1 = datetime.datetime.now()
        print(f'exiting {func.__name__} at {t1}')
        print(f'total time in {func.__name__}: {t1 - t0}')
        return x

    return wrapped_func


@timer
def model_fn(model_dir):
    return face_detection.build_detector(
        name=MODEL_NAME,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD)


def load_s3_image(bucket, key):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(key)[1]) as ntf:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        with open(ntf.name, 'wb') as fp:
            fp.write(obj['Body'].read())
        return cv2.imread(ntf.name)


@timer
def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    this is custom in that it doesn't do the pytorch tensor serialization here
    but instead lets the face_detect model handle necessary preprocessing in the
    predict step.

    additionally, we hijack the 'text/csv' and 'application/json' content_type
    modes:

        + content_type == 'text/csv': treat `input_data` as an s3 path
            `{bucket}/{key path}` and load the image directly using boto3 and
            cv2.
        + content_type == 'application/json': parse `input_data` for a `bucket`
            and a `key` attribute, and do the same as we do in CSV. if they are
            not found, try and parse the data with the sagemaker default
            `decoder.decode` method

    We lose the built-in functionality for CSV, but it is arguably the worst
    way to transfer the data anyway (npy is much better, and json at least
    allows the multi-dimensional arrays to stay multi-dimensional).

    note: application/json will also probably fail and receive a

        HTTP/1.1 413 Request Entity Too Large

    error (a standard size image as json is about 30M)

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns: numpy array.

    """
    if content_type == content_types.CSV:
        i = input_data.find('/')
        bucket, key = input_data[:i], input_data[i + 1:]
        # the [None] adds a dimension so it can be used in batched_detect
        return load_s3_image(bucket=bucket, key=key)[None]
    elif content_type == content_types.JSON:
        try:
            j = json.loads(input_data)
            # the [None] adds a dimension so it can be used in batched_detect
            return load_s3_image(bucket=j['bucket'], key=j['key'])[None]
        except (KeyError, TypeError):
            return decoder.decode(input_data, content_type)
    else:
        return decoder.decode(input_data, content_type)


@timer
def predict_fn(data, model):
    """custom prediction function

    Takes N RGB image and performs and returns a set of bounding boxes as
    detections

    Args:
        data: input data (np.ndarray) for prediction deserialized by input_fn
            should have shape [N, H, W, 3], where the 3 color channels are
            in RGB order. NOTE: cv2 loads images in BGR order.
        model: PyTorch model loaded in memory by model_fn

    Returns:
        boxes: list of length N with shape [num_boxes, 5] per element. the 5
            elements are (xmin, ymin, xmax, ymax, score)

    """
    if WITH_LANDMARKS:
        return model.batched_detect_with_landmarks(data)
    else:
        return model.batched_detect(data)
