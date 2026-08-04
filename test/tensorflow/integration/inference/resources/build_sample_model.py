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

    Layout: <output_dir>/<version>/saved_model.pb. Multi-version tuples
    archive side by side; TFS picks the highest by default. ``code_files``
    optionally installs user code under a top-level ``code/`` dir. Returns
    the absolute path to the produced tarball.
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
        # SM TFS/MME expects tarball -> <version>/saved_model.pb.
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
    """Return a tiny Conv2D Sequential model (117 params, deterministic init).

    Layers: Input(8,8,3) -> Conv2D(4, kernel=3, ReLU) -> GAP -> Dense(1),
    all trainable weights pinned to 1.0 and biases to 0.0. For an all-ones
    (1,8,8,3) input, closed-form output is 108.0 (see test_conv_gpu). A
    stubbed cuDNN kernel yields 0.0 — distinguishable, so the test fires.
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
    """Package <output_dir>/<version>/saved_model.pb into model.tar.gz."""
    tar_path = output_dir / tar_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(output_dir / str(version)), arcname=str(version))
    return str(tar_path)


def build_conv_sample_model(
    output_dir: str | os.PathLike | None = None,
    tar_filename: str = "model.tar.gz",
    version: int = 1,
) -> str:
    """Build a tiny Conv2D SavedModel via Keras 3 ``model.export()``.

    On GPU, TFS routes Conv2D through cuDNN's cudnnConvolutionForward —
    this SavedModel exists to exercise that path end-to-end at request time
    (sanity tests only check libcudnn presence, not ABI compatibility).
    """
    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-conv-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_dir / str(version)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = _build_conv_sequential()
    # Keras 3 dedicated SavedModel writer.
    model.export(str(saved_model_dir))

    return _package_saved_model_tarball(output_dir, version, tar_filename)


def _build_lstm_sequential():
    """Return a tiny LSTM Sequential model (units=1, deterministic init).

    Input(3, 2) -> LSTM(1, kernel=1, recurrent=1, bias=0). Constant weights
    make the forward pass deterministic — the same input on CPU and GPU
    yields the same output — so a cuDNN RNN regression (NaN, zero, or
    non-deterministic dispatch) is distinguishable at test time.

    LSTM is chosen over SimpleRNN because TF's cuDNN RNN dispatch fires
    specifically for LSTM / GRU with default (tanh + sigmoid) activations.
    """
    import tensorflow as tf

    kernel_ones = tf.keras.initializers.Constant(1.0)
    bias_zeros = tf.keras.initializers.Constant(0.0)

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3, 2), dtype=tf.float32),
            tf.keras.layers.LSTM(
                units=1,
                kernel_initializer=kernel_ones,
                recurrent_initializer=kernel_ones,
                bias_initializer=bias_zeros,
            ),
        ]
    )


def build_lstm_sample_model(
    output_dir: str | os.PathLike | None = None,
    tar_filename: str = "model.tar.gz",
    version: int = 1,
) -> str:
    """Build a tiny LSTM SavedModel via Keras 3 ``model.export()``.

    On GPU, TFS routes LSTM through cuDNN's RNN library (libcudnn_adv.so.9).
    Sanity tests assert the library file is present but no request exercises
    the RNN kernel until this. Constant-initialized weights + all-ones input
    give a deterministic output, so a regression in cuDNN RNN dispatch
    (NaN, zero, or run-to-run variance) is distinguishable.
    """
    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-lstm-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_dir / str(version)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = _build_lstm_sequential()
    # Keras 3 dedicated SavedModel writer.
    model.export(str(saved_model_dir))

    return _package_saved_model_tarball(output_dir, version, tar_filename)


if __name__ == "__main__":
    path = build_sample_model()
    print(path)
