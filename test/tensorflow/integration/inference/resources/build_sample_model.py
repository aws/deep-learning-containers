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


def build_conv_sample_model(
    output_dir: str | os.PathLike | None = None,
    tar_filename: str = "model.tar.gz",
    version: int = 1,
) -> str:
    """Build a tiny Conv2D SavedModel that exercises cuDNN kernels on GPU.

    Sanity tests only check ``libcudnn.so.*`` presence in ``ldconfig -p``
    and ``ldd tensorflow_model_server`` — they do NOT exercise a cuDNN op
    at request time. A cuDNN ABI drift (library present but incompatible
    with the TFS binary) would pass sanity and fault on the first customer
    Conv request. This model closes that gap: on GPU, TFS routes the
    ``Conv2D`` op through cuDNN's ``cudnnConvolutionForward`` path.

    Architecture (kept intentionally tiny — this is a smoke, not a
    benchmark):

    - Input: ``(None, 8, 8, 3)`` float32
    - ``Conv2D(4, kernel_size=3, activation='relu')`` -> 112 params
      (3*3*3*4 weights + 4 bias)
    - ``GlobalAveragePooling2D()`` -> 0 params
    - ``Dense(1)`` -> 5 params (4 weights + 1 bias)
    - Total: 117 trainable params

    Layout matches SageMaker TFS / MME: the tarball contains
    ``<version>/saved_model.pb`` at the top level so both single-model and
    multi-model endpoints can consume it without repackaging.
    """
    import tensorflow as tf

    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-conv-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    class ConvSmokeModel(tf.Module):
        """Small Conv2D SavedModel used to exercise cuDNN at request time.

        Wrapping the Keras model as an attribute on a ``tf.Module`` keeps the
        serving signature and the model's variables under the same trackable
        root, so TFS can bind every variable (e.g. ``dense/bias``) at invoke
        time. A bare ``tf.function`` that closes over the Keras model via
        Python scope leaves variables unowned by the SavedModel root and
        surfaces at request time as ``FAILED_PRECONDITION: Could not find
        variable dense/bias``.
        """

        def __init__(self, name: str = "conv_smoke") -> None:
            super().__init__(name=name)
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(8, 8, 3), dtype=tf.float32, name="input"),
                    tf.keras.layers.Conv2D(4, kernel_size=3, activation="relu", name="conv"),
                    tf.keras.layers.GlobalAveragePooling2D(name="gap"),
                    tf.keras.layers.Dense(1, name="dense"),
                ],
                name=name,
            )

        # Serving signature: fixed shape so TFS builds a deterministic input
        # spec that mirrors the payload the test sends.
        @tf.function(
            input_signature=[tf.TensorSpec(shape=[None, 8, 8, 3], dtype=tf.float32, name="input")]
        )
        def serve(self, x):
            return {"output": self.model(x, training=False)}

    module = ConvSmokeModel()

    saved_model_dir = output_dir / str(version)
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        module,
        str(saved_model_dir),
        signatures={"serving_default": module.serve},
    )

    tar_path = output_dir / tar_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(str(output_dir / str(version)), arcname=str(version))

    return str(tar_path)


if __name__ == "__main__":
    path = build_sample_model()
    print(path)
