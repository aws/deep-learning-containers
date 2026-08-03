"""Build a tiny TensorFlow SavedModel and tar it for SageMaker deployment.

The model performs ``y = x * multiplier`` for a runtime-supplied multiplier.
Tarballs are written to a caller-provided directory; nothing is checked in.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path


def build_sample_model(
    output_dir: str | os.PathLike | None = None,
    multiplier: float = 2.0,
    tar_filename: str = "model.tar.gz",
    versions: tuple[int, ...] = (1,),
    code_files: dict[str, str] | None = None,
) -> str:
    """Build a SavedModel that multiplies its input by ``multiplier``.

    The SavedModel is laid out under ``<output_dir>/<version>`` so that
    SageMaker's TensorFlow Serving handler can discover the version directory.
    Passing multiple version numbers (e.g. ``versions=(1, 2)``) writes each
    version's SavedModel side by side and archives all of them into the tarball;
    TensorFlow Serving selects the highest version by default, which lets tests
    exercise TFS's version-selection path.

    ``code_files`` optionally installs user code (e.g. ``inference.py`` with
    ``input_handler`` / ``output_handler`` / ``model_handler_load``) under a
    top-level ``code/`` directory in the tarball; the SageMaker TFS handler
    at ``python_service.py`` reads it from ``/opt/ml/model/code/`` (single
    model) or ``/opt/ml/models/<name>/model/code/`` (MME). Keys are relative
    paths inside ``code/``; values are the file contents. Returns the
    absolute path to the produced ``model.tar.gz``.
    """
    import tensorflow as tf

    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-sample-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    multiplier_const = tf.constant(multiplier, dtype=tf.float32)

    class MultiplierModel(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
        def __call__(self, x):
            return {"output": x * multiplier_const}

    model = MultiplierModel()

    for version in versions:
        # SageMaker TFS expects: model.tar.gz -> <version>/saved_model.pb
        # (SM MME expects numeric top-level version dir).
        saved_model_dir = output_dir / str(version)
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(
            model,
            str(saved_model_dir),
            signatures={"serving_default": model.__call__},
        )

    # Optional customer-supplied code files (e.g. inference.py).
    code_dir = output_dir / "code"
    if code_files:
        code_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, content in code_files.items():
            file_path = code_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

    tar_path = output_dir / tar_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        for version in versions:
            tar.add(str(output_dir / str(version)), arcname=str(version))
        if code_files:
            tar.add(str(code_dir), arcname="code")

    return str(tar_path)


def _build_conv_sequential():
    """Return the tiny Conv2D Sequential model used by the export builder.

    Architecture (kept intentionally tiny — this is a smoke, not a
    benchmark):

    - Input: ``(None, 8, 8, 3)`` float32
    - ``Conv2D(4, kernel_size=3, activation='relu')`` -> 112 params
      (3*3*3*4 weights + 4 bias)
    - ``GlobalAveragePooling2D()`` -> 0 params
    - ``Dense(1)`` -> 5 params (4 weights + 1 bias)
    - Total: 117 trainable params

    Closed-form deterministic init (B-1 fix). Every trainable parameter
    is pinned to a constant so the forward pass has a single correct
    numeric answer for an all-ones input; a stubbed / bypassed cuDNN
    kernel that returns a zero feature map produces a distinguishable
    output (``0.0``) that the test can catch loudly.

    Math (all-ones ``(1, 8, 8, 3)`` input, Conv2D kernel = 1.0 /
    bias = 0.0, Dense kernel = 1.0 / bias = 0.0):

    - Conv2D pre-activation at each of 6*6 spatial positions per filter:
      sum of a 3*3*3 patch of ones = ``27.0``. ReLU passes 27.0 through.
      Feature map shape ``(1, 6, 6, 4)``, every entry ``27.0``.
    - GAP averages the 6*6 spatial dims per filter -> ``(1, 4)`` of
      ``[27.0, 27.0, 27.0, 27.0]``.
    - Dense(1) with kernel=1.0, bias=0.0: 27+27+27+27 + 0 = ``108.0``.

    A dead / stubbed cuDNN convolution yields a zero feature map, which
    propagates to Dense output = 0.0 — the test's closed-form assertion
    (``scalar ≈ 108.0``) fires. This is why we no longer rely on
    random init + a Dense bias floor: with a bias of 1.0 alone, the
    Dense output for a zero feature map would be ``0 + 1.0 = 1.0``,
    which passed the old ``scalar != 0.0`` guard — a 100 % false
    negative on cuDNN kernel bypass.
    """
    import tensorflow as tf

    kernel_ones = tf.keras.initializers.Constant(1.0)
    bias_zeros = tf.keras.initializers.Constant(0.0)

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 3), dtype=tf.float32),
            tf.keras.layers.Conv2D(
                4,
                kernel_size=3,
                activation="relu",
                kernel_initializer=kernel_ones,
                bias_initializer=bias_zeros,
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(
                1,
                kernel_initializer=kernel_ones,
                bias_initializer=bias_zeros,
            ),
        ]
    )


def _package_saved_model_tarball(output_dir: Path, version: int, tar_filename: str) -> str:
    """Package ``<output_dir>/<version>/saved_model.pb`` into ``model.tar.gz``.

    SageMaker TFS / MME expects the tarball to contain a numeric version
    directory at the top level (``<version>/saved_model.pb``); the Conv
    builder below writes into that layout so this shared packager can
    finish the job.
    """
    tar_path = output_dir / tar_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(output_dir / str(version)), arcname=str(version))
    return str(tar_path)


def build_conv_sample_model(
    output_dir: str | os.PathLike | None = None,
    tar_filename: str = "model.tar.gz",
    version: int = 1,
) -> str:
    """Build a tiny Conv2D SavedModel using Keras 3 ``model.export()``.

    Modern customer path: customers train a Keras model and serialize with
    ``model.export(dir)`` — the Keras 3 dedicated SavedModel writer, which
    owns both the serving signature and the variable table so nothing gets
    orphaned during trace. This is what customers on TF >= 2.16 (Keras 3)
    are told to use.

    Sanity tests only check ``libcudnn.so.*`` presence in ``ldconfig -p``
    and ``ldd tensorflow_model_server`` — they do NOT exercise a cuDNN op
    at request time. A cuDNN ABI drift (library present but incompatible
    with the TFS binary) would pass sanity and fault on the first customer
    Conv request. This model closes that gap: on GPU, TFS routes the
    ``Conv2D`` op through cuDNN's ``cudnnConvolutionForward`` path.

    Layout matches SageMaker TFS / MME: the tarball contains
    ``<version>/saved_model.pb`` at the top level so both single-model and
    multi-model endpoints can consume it without repackaging.
    """
    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-conv-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_dir / str(version)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = _build_conv_sequential()
    # Keras 3 dedicated SavedModel writer. This is what customers use.
    model.export(str(saved_model_dir))

    return _package_saved_model_tarball(output_dir, version, tar_filename)


if __name__ == "__main__":
    path = build_sample_model()
    print(path)
