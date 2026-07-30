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
    """Return the tiny Conv2D Sequential model used by both export paths.

    Kept as a helper (not a fixture) so the modern (``model.export()``) and
    legacy (``tf.keras.models.save_model()``) builders below share exactly
    the same architecture. If they diverged, we'd lose the ability to say
    "same customer model, two export paths, one passes, one fails" — which
    is the entire point of the parametrized test that consumes these.

    Architecture (kept intentionally tiny — this is a smoke, not a
    benchmark):

    - Input: ``(None, 8, 8, 3)`` float32
    - ``Conv2D(4, kernel_size=3, activation='relu')`` -> 112 params
      (3*3*3*4 weights + 4 bias)
    - ``GlobalAveragePooling2D()`` -> 0 params
    - ``Dense(1)`` -> 5 params (4 weights + 1 bias)
    - Total: 117 trainable params
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 3), dtype=tf.float32),
            tf.keras.layers.Conv2D(4, kernel_size=3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1),
        ]
    )


def _package_saved_model_tarball(output_dir: Path, version: int, tar_filename: str) -> str:
    """Package ``<output_dir>/<version>/saved_model.pb`` into ``model.tar.gz``.

    SageMaker TFS / MME expects the tarball to contain a numeric version
    directory at the top level (``<version>/saved_model.pb``). The two Conv
    builders below both write into that layout so this shared packager can
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


def build_conv_sample_model_legacy_save(
    output_dir: str | os.PathLike | None = None,
    tar_filename: str = "model.tar.gz",
    version: int = 1,
) -> str:
    """Build a tiny Conv2D SavedModel using the legacy Keras save path.

    Legacy customer path: pre-Keras-3 customers (or customers with existing
    training scripts they haven't migrated) save via
    ``tf.keras.models.save_model(model, dir, save_format="tf")`` or
    ``model.save(dir, save_format="tf")``. Keras 3 has been progressively
    tightening what ``save_format="tf"`` accepts (the default is the
    ``.keras`` zip format), but the SavedModel path is still the migration
    surface for most existing customer code and must keep working on TFS.

    We try ``tf.keras.models.save_model(..., save_format="tf")`` first, then
    fall back to ``model.save(..., save_format="tf")``. If both raise, we
    re-raise the last error rather than fake success — a real regression in
    Keras 3 rejecting SavedModel from ``save_model`` is exactly what this
    test exists to catch.

    Same architecture and tarball layout as ``build_conv_sample_model``.
    """
    import tensorflow as tf

    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-conv-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_dir / str(version)
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    model = _build_conv_sequential()

    # Preferred legacy path: tf.keras.models.save_model(..., save_format="tf").
    # Fallback: model.save(..., save_format="tf") — historically the same
    # writer, but Keras 3 has been picky about whether save_model routes
    # SavedModel or requires a .keras extension.
    try:
        tf.keras.models.save_model(model, str(saved_model_dir), save_format="tf")
    except (ValueError, TypeError) as save_model_err:
        try:
            model.save(str(saved_model_dir), save_format="tf")
        except (ValueError, TypeError) as save_err:
            raise RuntimeError(
                "Neither tf.keras.models.save_model(..., save_format='tf') nor "
                "model.save(..., save_format='tf') accepted the SavedModel path. "
                f"save_model: {save_model_err!r}; save: {save_err!r}"
            ) from save_err

    return _package_saved_model_tarball(output_dir, version, tar_filename)


if __name__ == "__main__":
    path = build_sample_model()
    print(path)
