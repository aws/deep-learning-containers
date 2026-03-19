# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Shared test helpers, constants, and test data for Ray serve tests (EC2 and SageMaker)."""

import io
import math
import os
import struct
import wave
import zlib
from urllib.request import urlretrieve

# Public S3 test images
TEST_IMAGES = {
    "kitten": "https://s3.amazonaws.com/model-server/inputs/kitten.jpg",
    "flower": "https://s3.amazonaws.com/model-server/inputs/flower.jpg",
}

# S3 location for model tarballs
S3_BUCKET = "dlc-cicd-models"
S3_PREFIX = "rayserve-models"

MIN_MNIST_ACCURACY = 80  # percent

# Iris samples: (features, expected_species, description)
IRIS_SAMPLES = [
    ([5.1, 3.5, 1.4, 0.2], "setosa", "Setosa sample 1"),
    ([4.9, 3.0, 1.4, 0.2], "setosa", "Setosa sample 2"),
    ([6.4, 3.2, 4.5, 1.5], "versicolor", "Versicolor sample 1"),
    ([5.7, 2.8, 4.1, 1.3], "versicolor", "Versicolor sample 2"),
    ([6.3, 3.3, 6.0, 2.5], "virginica", "Virginica sample 1"),
    ([7.2, 3.6, 6.1, 2.5], "virginica", "Virginica sample 2"),
]

# Sentiment samples: (text, expected_label)
SENTIMENT_SAMPLES = [
    ("This product is absolutely amazing!", "POSITIVE"),
    ("I love this so much, best purchase ever!", "POSITIVE"),
    ("This is terrible and broken", "NEGATIVE"),
    ("Worst experience of my life", "NEGATIVE"),
    ("This is awful, a complete waste of money", "NEGATIVE"),
    ("Absolutely perfect, highly recommend!", "POSITIVE"),
]


def download_test_image(url, path):
    """Download a test image if not already cached."""
    if not os.path.exists(path):
        urlretrieve(url, path)
    with open(path, "rb") as f:
        return f.read()


def download_all_test_images(cache_dir="/tmp/ray_test_images"):
    """Download all test images, return dict of {name: bytes}."""
    os.makedirs(cache_dir, exist_ok=True)
    images = {}
    for name, url in TEST_IMAGES.items():
        path = os.path.join(cache_dir, f"{name}.jpg")
        images[name] = download_test_image(url, path)
    return images


def make_sine_wav(freq=440, duration=0.5, sample_rate=16000):
    """Generate a sine wave WAV as bytes."""
    n_samples = int(sample_rate * duration)
    samples = [
        int(math.sin(2 * math.pi * freq * i / sample_rate) * 32767) for i in range(n_samples)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def make_all_sine_wavs():
    """Generate 3 sine waves at different frequencies (440, 880, 220 Hz).

    Returns list of (name, bytes) tuples.
    """
    wavs = []
    for freq in [440, 880, 220]:
        wavs.append((f"sine_{freq}hz", make_sine_wav(freq=freq, duration=0.5)))
    return wavs


def _make_digit_pixels(digit_id):
    """Generate 28x28 grayscale pixel array for a digit (0-9)."""
    d = [0] * 784

    if digit_id == 0:
        for y in range(28):
            for x in range(28):
                cy, cx = y - 14, x - 14
                dist = (cy**2 / 100 + cx**2 / 49) ** 0.5
                if 0.6 < dist < 1.0:
                    d[y * 28 + x] = 255
    elif digit_id == 1:
        for y in range(4, 24):
            for x in range(12, 16):
                d[y * 28 + x] = 255
    elif digit_id == 2:
        for x in range(8, 20):
            for dy in range(3):
                d[(4 + dy) * 28 + x] = 255
                d[(22 + dy) * 28 + x] = 255
        for i in range(18):
            y = 4 + i
            x = 19 - i * 11 // 18
            if 0 <= x < 28 and 0 <= y < 28:
                d[y * 28 + x] = 255
                if x + 1 < 28:
                    d[y * 28 + x + 1] = 255
    elif digit_id == 3:
        for x in range(8, 20):
            for dy in range(3):
                d[(4 + dy) * 28 + x] = 255
                d[(12 + dy) * 28 + x] = 255
                d[(22 + dy) * 28 + x] = 255
        for y in range(4, 25):
            for dx in range(3):
                d[y * 28 + 18 + dx] = 255
    elif digit_id == 4:
        for y in range(4, 15):
            d[y * 28 + 8] = 255
            d[y * 28 + 9] = 255
        for x in range(8, 21):
            d[14 * 28 + x] = 255
            d[15 * 28 + x] = 255
        for y in range(4, 24):
            d[y * 28 + 18] = 255
            d[y * 28 + 19] = 255
    elif digit_id == 5:
        for x in range(8, 20):
            for dy in range(3):
                d[(4 + dy) * 28 + x] = 255
                d[(13 + dy) * 28 + x] = 255
                d[(22 + dy) * 28 + x] = 255
        for y in range(4, 15):
            d[y * 28 + 8] = 255
            d[y * 28 + 9] = 255
        for y in range(13, 25):
            d[y * 28 + 18] = 255
            d[y * 28 + 19] = 255
    elif digit_id == 6:
        for y in range(28):
            for x in range(28):
                cy, cx = y - 9, x - 14
                dist_top = (cy**2 / 25 + cx**2 / 25) ** 0.5
                if 0.6 < dist_top < 1.0 and x <= 14:
                    d[y * 28 + x] = 255
                cy2 = y - 19
                dist_bot = (cy2**2 / 25 + cx**2 / 25) ** 0.5
                if 0.6 < dist_bot < 1.0:
                    d[y * 28 + x] = 255
    elif digit_id == 7:
        for x in range(8, 21):
            d[4 * 28 + x] = 255
            d[5 * 28 + x] = 255
        for i in range(20):
            y = 5 + i
            x = 20 - i * 6 // 20
            if 0 <= y < 28 and 0 <= x < 28:
                d[y * 28 + x] = 255
                if x + 1 < 28:
                    d[y * 28 + x + 1] = 255
    elif digit_id == 8:
        for y in range(28):
            for x in range(28):
                cy1, cy2 = y - 9, y - 19
                cx = x - 14
                d1 = (cy1**2 / 25 + cx**2 / 25) ** 0.5
                d2 = (cy2**2 / 25 + cx**2 / 25) ** 0.5
                if (0.6 < d1 < 1.0) or (0.6 < d2 < 1.0):
                    d[y * 28 + x] = 255
    elif digit_id == 9:
        for y in range(28):
            for x in range(28):
                cy, cx = y - 9, x - 14
                dist = (cy**2 / 25 + cx**2 / 25) ** 0.5
                if 0.6 < dist < 1.0:
                    d[y * 28 + x] = 255
        for y in range(9, 25):
            d[y * 28 + 19] = 255
            d[y * 28 + 20] = 255
    else:
        # Fallback: vertical bar offset by digit_id
        col = 6 + digit_id
        for y in range(4, 24):
            for x in range(col, col + 3):
                if 0 <= x < 28:
                    d[y * 28 + x] = 255
    return d


def make_digit_png(digit_id):
    """Generate a minimal 28x28 grayscale PNG for a digit (0-9)."""
    width, height = 28, 28
    pixels = _make_digit_pixels(digit_id)

    raw = b""
    for y in range(height):
        raw += b"\x00"
        for x in range(width):
            raw += bytes([pixels[y * width + x]])

    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def make_all_digit_pngs():
    """Generate images for digits 0-9. Returns dict of {digit_id: png_bytes}."""
    return {d: make_digit_png(d) for d in range(10)}


# ---------------------------------------------------------------------------
# Response validators
# ---------------------------------------------------------------------------


def validate_densenet_response(result):
    """Validate DenseNet response: predictions list with class_id, class_name, probability."""
    if "predictions" not in result:
        return "Missing 'predictions' field"
    preds = result["predictions"]
    # DenseNet deployment returns exactly top-5 predictions
    if not isinstance(preds, list) or len(preds) != 5:
        return f"Expected 5 predictions, got {type(preds).__name__} len={len(preds) if isinstance(preds, list) else 'N/A'}"
    for i, p in enumerate(preds):
        for key in ("class_id", "class_name", "probability"):
            if key not in p:
                return f"predictions[{i}] missing '{key}'"
        if not isinstance(p["class_id"], int):
            return f"predictions[{i}].class_id not int"
        if p["probability"] <= 0:
            return f"predictions[{i}].probability <= 0"
    return ""


def validate_mnist_response(result):
    """Validate MNIST response: prediction (int 0-9), confidence, probabilities."""
    for key in ("prediction", "confidence", "probabilities"):
        if key not in result:
            return f"Missing '{key}' field"
    if not isinstance(result["prediction"], int):
        return "prediction not int"
    if not (0 <= result["prediction"] <= 9):
        return f"prediction {result['prediction']} out of range"
    if not isinstance(result["probabilities"], dict):
        return "probabilities must be dict"
    if result["confidence"] <= 0:
        return "confidence <= 0"
    return ""


def validate_iris_response(result):
    """Validate Iris tabular response: prediction species, confidence, probabilities."""
    for key in ("prediction", "confidence", "probabilities"):
        if key not in result:
            return f"Missing '{key}' field"
    if result["prediction"] not in ("setosa", "versicolor", "virginica"):
        return f"Unknown prediction: {result['prediction']}"
    if result["confidence"] <= 0:
        return "confidence <= 0"
    return ""


def validate_sentiment_response(result):
    """Validate NLP sentiment response: predictions list with label and score."""
    if "predictions" not in result:
        return "Missing 'predictions' field"
    preds = result["predictions"]
    if not isinstance(preds, list) or len(preds) == 0:
        return f"Expected non-empty predictions list, got {type(preds).__name__}"
    for i, p in enumerate(preds):
        if "label" not in p:
            return f"predictions[{i}] missing 'label'"
        if "score" not in p:
            return f"predictions[{i}] missing 'score'"
        if p["label"] not in ("POSITIVE", "NEGATIVE"):
            return f"predictions[{i}] unknown label: {p['label']}"
        if p["score"] <= 0:
            return f"predictions[{i}] score <= 0"
    return ""


def validate_audio_response(result, check_ffmpeg_backend=False):
    """Validate audio transcription response."""
    if "transcription" not in result:
        return "Missing 'transcription' field"
    if not isinstance(result["transcription"], str):
        return "transcription is not a string"
    if check_ffmpeg_backend:
        backend = result.get("audio_backend", "MISSING")
        if backend != "ffmpeg":
            return f"Expected audio_backend='ffmpeg', got '{backend}'"
    return ""
