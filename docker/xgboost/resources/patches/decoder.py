# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This module contains functionality for converting various types of
files and objects to NumPy arrays."""
from __future__ import absolute_import

import json

import numpy as np
import scipy.sparse
from six import BytesIO, StringIO

from sagemaker_inference import content_types, errors


def _json_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a JSON object to a numpy array.

        Args:
            string_like (str): JSON string.
            dtype (dtype, optional):  Data type of the resulting array.
                If None, the dtypes will be determined by the contents
                of each column, individually. This argument can only be
                used to 'upcast' the array.  For downcasting, use the
                .astype(t) method.

        Returns:
            (np.array): numpy array
        """
    data = json.loads(string_like)
    return np.array(data, dtype=dtype)


def _csv_to_numpy(string_like, dtype=None):  # type: (str) -> np.array
    """Convert a CSV object to a numpy array.

    Args:
        string_like (str): CSV string.
        dtype (dtype, optional):  Data type of the resulting array. If None,
            the dtypes will be determined by the contents of each column,
            individually. This argument can only be used to 'upcast' the array.
            For downcasting, use the .astype(t) method.

    Returns:
        (np.array): numpy array
    """
    stream = StringIO(string_like)
    return np.genfromtxt(stream, dtype=dtype, delimiter=",")


def _npy_to_numpy(npy_array):  # type: (object) -> np.array
    """Convert a NPY array into numpy.

    Args:
        npy_array (npy array): to be converted to numpy array

    Returns:
        (np.array): converted numpy array.
    """
    stream = BytesIO(npy_array)
    return np.load(stream, allow_pickle=False)


def _npz_to_sparse(npz_bytes):  # type: (object) -> scipy.sparse.spmatrix
    """Convert .npz-formatted data to a sparse matrix.

    Args:
        npz_bytes (object): Bytes encoding a sparse matrix in the .npz format.

    Returns:
        (scipy.sparse.spmatrix): A sparse matrix.
    """
    buffer = BytesIO(npz_bytes)
    return scipy.sparse.load_npz(buffer)


_decoder_map = {
    content_types.NPY: _npy_to_numpy,
    content_types.CSV: _csv_to_numpy,
    content_types.JSON: _json_to_numpy,
    content_types.NPZ: _npz_to_sparse,
}


def decode(obj, content_type):
    """Decode an object that is encoded as one of the default content types.

    Args:
        obj (object): to be decoded.
        content_type (str): content type to be used.

    Returns:
        object: decoded object for prediction.
    """
    try:
        decoder = _decoder_map[content_type]
        return decoder(obj)
    except KeyError:
        raise errors.UnsupportedFormatError(content_type)
