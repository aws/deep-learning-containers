#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: inference
Created: 2020-08-19

Description:

    inference functions for pytorch serving container

"""
import datetime
import json
import os
import sys
import tempfile

import boto3
import torch
import yaml

from sagemaker_inference import content_types, errors

# path munging to access the neighboring YAEDP repo
HERE = os.path.dirname(os.path.realpath(__file__))
YAEDP_ROOT = os.path.join(HERE, 'Yet-Another-EfficientDet-Pytorch')
sys.path.insert(0, YAEDP_ROOT)

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess, preprocess

# model version is chosen at model startup from the invocation arguments
EFFICIENTDET_COMPOUND_COEF = int(os.environ.get('EFFICIENTDET_COMPOUND_COEF', 0))
DEFAULT_PRED_THRESHOLD = float(os.environ.get('DEFAULT_PRED_THRESHOLD', 0.05))
DEFAULT_IOU_THRESHOLD = float(os.environ.get('DEFAULT_IOU_THRESHOLD', 0.5))
USE_FLOAT16 = os.environ.get('USE_FLOAT16', 'false') == 'true'

print(f"EFFICIENTDET_COMPOUND_COEF = {EFFICIENTDET_COMPOUND_COEF}")
print(f"DEFAULT_PRED_THRESHOLD = {DEFAULT_PRED_THRESHOLD}")
print(f"DEFAULT_IOU_THRESHOLD = {DEFAULT_IOU_THRESHOLD}")
print(f"USE_FLOAT16 = {USE_FLOAT16}")

USE_CUDA = torch.cuda.is_available()

print(f'USE_CUDA = {USE_CUDA}')

# pre and post processing params from YAEDP
with open(os.path.join(YAEDP_ROOT, 'projects', 'coco.yml')) as fp:
    PARAMS = yaml.safe_load(fp)

print(f'PARAMS = {PARAMS}')

# input size is fixed on the compound coef level
MAX_INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
MAX_INPUT_SIZE = MAX_INPUT_SIZES[EFFICIENTDET_COMPOUND_COEF]
print(f"MAX_INPUT_SIZE = {MAX_INPUT_SIZE}")


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


class StoryblocksCustomError(errors.GenericInferenceToolkitError):
    def __init__(self, message):
        super().__init__(400, message)


def get_weights_url(c):
    """this list of urls is hand-created from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/readme.md as of the
    time of writing this file (2020-12-03). as the model is updated, these urls *could* (but
    likely won't) change much. of course we may need to update this over time

    this could be automated with the github releases api, but that's overkill rn
    https://developer.github.com/v3/repos/releases/

    Args
        c: the compound coefficient (read the efficientnet/det papers for details)

    Returns
        url of the weights file to download

    """
    url_map = {
        '0': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d0.pth',
        '1': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d1.pth',
        '2': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d2.pth',
        '3': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d3.pth',
        '4': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d4.pth',
        '5': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d5.pth',
        '6': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
             '/efficientdet-d6.pth',
        '7': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
             '/efficientdet-d7.pth',
        '8': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
             '/efficientdet-d8.pth',
        '7x': 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
              '/efficientdet-d8.pth', }

    c = str(c)
    try:
        return url_map[c]
    except KeyError:
        raise StoryblocksCustomError(
            "compound coefficient not found, cannot download model weights")


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
    # based entirely off of
    # https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/coco_eval.py
    print(f'building and loading efficientdet d{EFFICIENTDET_COMPOUND_COEF}')
    model = EfficientDetBackbone(compound_coef=EFFICIENTDET_COMPOUND_COEF,
                                 num_classes=len(PARAMS['obj_list']),
                                 ratios=eval(PARAMS['anchors_ratios']),
                                 scales=eval(PARAMS['anchors_scales']))
    state_dict = torch.hub.load_state_dict_from_url(
        url=get_weights_url(c=EFFICIENTDET_COMPOUND_COEF),
        model_dir=model_dir,
        map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()

    if USE_CUDA:
        model.cuda(0)

    if USE_FLOAT16:
        model.half()

    return model


def load_s3_image_and_preprocess(bucket, key):
    """download the file from s3, process it using the external library

    returns:
        ori_imgs (list of np.ndarray) the original image as a cv2 np array
        framed_imgs (list of np.ndarray) a resized and rescaled version of each image in ori_imgs
        framed_metas (list of tuples) a list of tuples of scaling information, where each tuple
            has components new_w, new_h, old_w, old_h, padding_w, padding_h

    """
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(key)[1]) as ntf:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        with open(ntf.name, 'wb') as fp:
            fp.write(obj['Body'].read())

        return preprocess(ntf.name, max_size=MAX_INPUT_SIZE, mean=PARAMS['mean'], std=PARAMS['std'])


def parse_json_input(input_data):
    try:
        j = json.loads(input_data)
        bucket = j['bucket']
        key = j['key']
        pred_threshold = j.get('pred_threshold', DEFAULT_PRED_THRESHOLD)
        assert 0 <= pred_threshold <= 1
        iou_threshold = j.get('iou_threshold', DEFAULT_IOU_THRESHOLD)
        assert 0 <= iou_threshold <= 1
        return bucket, key, pred_threshold, iou_threshold
    except (KeyError, TypeError):
        raise StoryblocksCustomError(
            f"payloads for content type {content_types.JSON} must have json bodies with keys "
            f"\"key\" and \"bucket\"")
    except AssertionError:
        raise StoryblocksCustomError(
            "provided threshold values must be between 0 and 1 (inclusive)")


@timer
def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    this is custom in that it doesn't do the pytorch tensor serialization here but instead lets
    the Yet-Another-EfficientDet-Pytorch module handle necessary preprocessing in the predict step.

    additionally, we hijack the 'text/csv' and 'application/json' content_type modes:

        + content_type == 'text/csv': treat `input_data` as an s3 path `{bucket}/{key path}` and
            load the image directly using boto3 and cv2.
        + content_type == 'application/json': parse `input_data` for a `bucket` and a `key`
            attribute, and do the same as we do in CSV. additionally allows users to parameterize
            the prediction threshold and iou threshold values with keys `pred_threshold` and
            `iou_threshold`

    We lose the built-in functionality for CSV, but it is arguably the worst way to transfer the
    data anyway (npy is much better, and json at least allows the multi-dimensional arrays to
    stay multi-dimensional).

    note: application/json will also probably fail and receive a

        HTTP/1.1 413 Request Entity Too Large

    error (a standard size image as json is about 30M)

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
    # todo: can we support multiple input files?
    if content_type == content_types.CSV:
        i = input_data.find('/')
        bucket, key = input_data[:i], input_data[i + 1:]
        pred_threshold = DEFAULT_PRED_THRESHOLD
        iou_threshold = DEFAULT_IOU_THRESHOLD
    elif content_type == content_types.JSON:
        bucket, key, pred_threshold, iou_threshold = parse_json_input(input_data)
    else:
        raise errors.UnsupportedFormatError(content_type)

    ori_imgs, framed_imgs, framed_metas = load_s3_image_and_preprocess(bucket=bucket, key=key)
    return ori_imgs, framed_imgs, framed_metas, pred_threshold, iou_threshold


@timer
def predict_fn(data, model):
    """mostly copied from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet_test.py

    Args:
        data: tuple of inputs generated by custom input_fn above
        model: PyTorch model loaded in memory by model_fn

    Returns: a prediction

    """
    ori_imgs, framed_imgs, framed_metas, threshold, iou_threshold = data
    x = torch.stack(
        [torch.from_numpy(fi).cuda() if USE_CUDA else torch.from_numpy(fi)
         for fi in framed_imgs],
        0)

    x = x.to(torch.float32 if not USE_FLOAT16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regress_boxes = BBoxTransform()
        clip_boxes = ClipBoxes()

        out = postprocess(x,
                          anchors=anchors,
                          regression=regression,
                          classification=classification,
                          regressBoxes=regress_boxes,
                          clipBoxes=clip_boxes,
                          threshold=threshold,
                          iou_threshold=iou_threshold)
        out = invert_affine(framed_metas, out)
        return out
