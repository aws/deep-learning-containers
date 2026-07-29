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


if __name__ == "__main__":
    path = build_sample_model()
    print(path)
