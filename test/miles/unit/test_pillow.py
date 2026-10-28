"""Verify Pillow, which decodes the images that multimodal rollouts read."""

import io

import numpy as np
from PIL import Image


def test_jpeg_round_trip_preserves_the_image():
    rows, cols = np.mgrid[0:64, 0:48]
    pixels = np.stack([rows * 4, cols * 5, (rows + cols) * 2], axis=-1).astype(np.uint8)
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="JPEG", quality=95)

    decoded = Image.open(io.BytesIO(encoded.getvalue()))
    assert decoded.format == "JPEG"
    decoded_pixels = np.asarray(decoded.convert("RGB"))
    assert decoded_pixels.shape == pixels.shape
    assert np.abs(decoded_pixels.astype(int) - pixels).mean() < 4
