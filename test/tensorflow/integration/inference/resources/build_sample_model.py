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
) -> str:
    """Build a SavedModel that multiplies its input by ``multiplier``.

    The SavedModel is laid out under ``<output_dir>/export/Servo/1`` so that
    SageMaker's TensorFlow Serving handler can discover the version directory.
    Returns the absolute path to the produced ``model.tar.gz``.
    """
    import tensorflow as tf

    output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="tf220-sample-"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # SageMaker TFS expects: model.tar.gz -> <version>/saved_model.pb  (SM MME expects numeric top-level version dir)
    saved_model_dir = output_dir / "1"
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    multiplier_const = tf.constant(multiplier, dtype=tf.float32)

    class MultiplierModel(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
        def __call__(self, x):
            return {"output": x * multiplier_const}

    model = MultiplierModel()
    tf.saved_model.save(
        model,
        str(saved_model_dir),
        signatures={"serving_default": model.__call__},
    )

    tar_path = output_dir / tar_filename
    with tarfile.open(tar_path, "w:gz") as tar:
        # Archive the numeric version dir as the top-level entry inside the tarball.
        tar.add(str(output_dir / "1"), arcname="1")

    return str(tar_path)


if __name__ == "__main__":
    path = build_sample_model()
    print(path)
