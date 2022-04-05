#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module:
Created: 2022-03-30

Description:

    inference functions for pytorch serving container for topic translation

    yes this is insane overkill for what is basically just a simple matrix multiplication but hey
    whatcha gonna do this is the framework all our other models are in so

"""
import datetime
import json
import logging
import os

import boto3
import numpy as np
from sagemaker_inference import content_types, errors
from scipy.sparse import dok_matrix

LOGGER = logging.getLogger(__name__)

# supported environment variables and defaults
SRC_CLASS = os.getenv('SRC_CLASS', 'video')
TGT_CLASS = os.getenv('TGT_CLASS', 'audio')
TTL_SECONDS = int(os.getenv('TTL_SECONDS', 4 * 60 * 60))

LOGGER.info(f"SRC_CLASS = {SRC_CLASS}")
LOGGER.info(f"TGT_CLASS = {TGT_CLASS}")
LOGGER.info(f"TTL_SECONDS = {TTL_SECONDS}")

# weight arrays are cached in a globally accessible singleton map (note: not shared across workers,
# obviously, but within workers)
TRANSLATION_ARRAYS = {}


class StoryblocksCustomError(errors.GenericInferenceToolkitError):
    def __init__(self, message):
        super().__init__(400, message)


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
def load_array(src_type: str = 'footage', tgt_type: str = 'music') -> None:
    """attempt to read a topic translation array from s3

    Args:
        src_type: the content type of the input source vector
        tgt_type: the content type of the output translated vector
    """
    global TRANSLATION_ARRAYS
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('videoblocks-ml')
    basename = f'{src_type}_{tgt_type}.npy'
    key = f'models/topic-tran/storyblocks/prod/{SRC_CLASS}-{TGT_CLASS}-arrays/{basename}'
    obj = bucket.Object(key=key)
    f_local = f'/tmp/{basename}'
    obj.download_file(Filename=f_local)
    TRANSLATION_ARRAYS[src_type, tgt_type] = {'array': np.load(f_local),
                                              't': datetime.datetime.now(datetime.timezone.utc)}


def refresh_array_and_return(src_type: str = 'footage', tgt_type: str = 'music'):
    LOGGER.info(f"array for ({src_type}, {tgt_type}) hasn't been loaded or is expired. reloading")
    load_array(src_type=src_type, tgt_type=tgt_type)
    return TRANSLATION_ARRAYS[src_type, tgt_type]['array']


def get_translation_array(src_type: str = 'footage', tgt_type: str = 'music'):
    """implements ttl cache for array lookup / loading and therefore allows us to update translation
    arrays in prod with some defined frequency

    Args:
        src_type: the content type of the input source vector
        tgt_type: the content type of the output translated vector
    """
    global TRANSLATION_ARRAYS

    try:
        translation_array_dict = TRANSLATION_ARRAYS[src_type, tgt_type]
    except KeyError:
        return refresh_array_and_return(src_type=src_type, tgt_type=tgt_type)

    now = datetime.datetime.now(datetime.timezone.utc)
    array_age_in_secs = (now - translation_array_dict['t']).total_seconds()
    if array_age_in_secs < TTL_SECONDS:
        return translation_array_dict['array']
    else:
        return refresh_array_and_return(src_type=src_type, tgt_type=tgt_type)


def parse_json_input(input_data):
    try:
        j = json.loads(input_data)
        src_content_type = j['srcContentType']
        tgt_content_type = j['tgtContentType']
        dim = j['sparseVector']['dim']
        vector_dict = j['sparseVector']['vector']

        vector = np.zeros(dim)
        for (idx_str, val) in vector_dict.items():
            idx = int(idx_str)
            vector[idx] = val

        return src_content_type, tgt_content_type, vector
    except (KeyError, TypeError):
        raise StoryblocksCustomError(
            f"payloads for content type {content_types.JSON} must have json bodies with keys "
            f"\"srcContentType\", \"tgtContentType\", and \"sparseVector\". futhermore, the value "
            f"for \"sparseVector\" must be an object with keys \"dim\" and \"vector\"")
    except IndexError:
        raise StoryblocksCustomError(
            "sparseVector.vector contains keys greater that sparseVector.dim")


@timer
def model_fn(model_dir):
    return None


@timer
def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats. parse all into a dense
    topic vector

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns:
        ori_imgs (list of np.ndarray) the original image as a cv2 np array
        framed_imgs (list of np.ndarray) a resized and rescaled version of each image in ori_imgs
        framed_metas (list of tuples) a list of tuples of scaling information, where each tuple
            has components new_w, new_h, old_w, old_h, padding_w, padding_h
        pred_threshold (float): threshold for filtering detection predictions
        iou_threshold (float): threshold for iou filtering in nms postprocessing of detections

    """
    if content_type == content_types.JSON:
        return parse_json_input(input_data)
    else:
        raise errors.UnsupportedFormatError(content_type)


@timer
def predict_fn(data, model):
    """just matrix multiplication, y'all

    Args:
        data: tuple of inputs generated by custom input_fn above
        model: PyTorch model loaded in memory by model_fn

    Returns: a vector translated into the desired content type

    """
    src_type, tgt_type, vector = data
    translation_array = get_translation_array(src_type=src_type, tgt_type=tgt_type)
    try:
        dense_output = translation_array @ vector
    except ValueError:
        raise StoryblocksCustomError(
            f"input sparse vector dimension does not match dimensions for srcContentType = "
            f"{src_type} ({translation_array.shape[1]}")
    out_dim = dense_output.shape[0]
    dok_output = dok_matrix(dense_output.reshape(1, out_dim))
    return {'dim': out_dim,
            'vector': {str(idx): val for ((_, idx), val) in dok_output.items()}}
