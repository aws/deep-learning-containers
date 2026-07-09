#!/usr/bin/env python3
"""Basic sanity checks for the RayTrain DLC.

Import-and-version level only — runs on a CPU sanity runner, so this does NOT
exercise GPU/CUDA/NCCL/EFA (that is the deferred multi-node EFA/EKS test tier).
It verifies the training stack is present, importable, and pinned as expected.

Env:
  EXPECTED_FRAMEWORK_VERSION — expected Ray version (e.g. "2.56.0")
"""

import os
import sys


def _check_imports():
    import accelerate  # noqa: F401
    import datasets  # noqa: F401
    import pytorch_lightning  # noqa: F401
    import ray  # noqa: F401
    import ray.data  # noqa: F401
    import ray.train  # noqa: F401
    import ray.tune  # noqa: F401
    import torch  # noqa: F401
    import torchvision  # noqa: F401
    import transformers  # noqa: F401

    print(
        "imports OK: ray[train,tune,data], torch, torchvision, lightning, "
        "transformers, datasets, accelerate"
    )


def _check_versions():
    import ray
    import torch

    print(f"ray={ray.__version__} torch={torch.__version__}")

    expected_ray = os.environ.get("EXPECTED_FRAMEWORK_VERSION", "")
    if expected_ray and ray.__version__ != expected_ray:
        raise AssertionError(
            f"ray version mismatch: expected {expected_ray}, got {ray.__version__}"
        )

    # torch must be a CUDA build (cuXXX in the local version), even though we
    # can't run CUDA on the CPU sanity runner.
    if "cu" not in torch.__version__:
        raise AssertionError(f"torch is not a CUDA build (version={torch.__version__})")


def _check_serve_absent():
    """RayTrain is training-scoped — the Ray Serve extra must NOT be installed.

    The `ray.serve` module always ships inside the ray package, so importing it
    is the real signal: without the `serve` extra its deps (e.g. starlette) are
    missing and the import fails. A successful import means serve deps leaked in.
    """
    try:
        import ray.serve  # noqa: F401
    except ImportError:
        print("ray.serve extra correctly absent (training-scoped image)")
        return
    raise AssertionError(
        "ray.serve imported successfully — its dependencies are installed, but "
        "RayTrain must be training-scoped (use the Ray Serve DLC for inference)"
    )


def main():
    try:
        _check_imports()
        _check_versions()
        _check_serve_absent()
    except Exception as exc:  # noqa: BLE001
        print(f"RayTrain sanity FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    print("RayTrain sanity PASSED")


if __name__ == "__main__":
    main()
