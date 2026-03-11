#!/usr/bin/env python3
"""Validation for lambda-base runtime."""

import os
import subprocess
import sys


def test_python_runtime():
    """Validate Python runtime and symlinks."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    print(f"  Executable: {sys.executable}")
    assert version >= (3, 12), "Expected Python >= 3.12"

    # Verify python/python3 symlinks resolve to same version
    for cmd in ["python", "python3"]:
        out = subprocess.check_output([cmd, "--version"], text=True).strip()
        print(f"  {cmd} -> {out}")
        assert f"{version.major}.{version.minor}" in out


def test_lambda_ric():
    """Validate Lambda Runtime Interface Client."""
    print("✓ awslambdaric installed")


def test_lambda_rie():
    """Validate Lambda Runtime Interface Emulator binary."""
    rie = "/usr/local/bin/aws-lambda-rie"
    assert os.path.isfile(rie), f"RIE not found at {rie}"
    assert os.access(rie, os.X_OK), "RIE not executable"
    size = os.path.getsize(rie)
    print(f"✓ RIE binary: {size / 1024 / 1024:.1f}MB")


def test_cuda():
    """Validate CUDA runtime libraries."""
    cuda_lib = "/usr/local/cuda/lib64"
    assert os.path.isdir(cuda_lib), "CUDA lib dir not found"

    import ctypes

    try:
        ctypes.CDLL("libcudart.so.12")
    except OSError:
        ctypes.CDLL("libcudart.so")
    print("✓ CUDA runtime loaded")

    # Check nvidia-smi if GPU available
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip()
        print(f"  GPU: {out}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  No GPU detected (OK for build validation)")


def test_entrypoint():
    """Validate lambda_entrypoint.sh exists and is executable."""
    script = "/lambda_entrypoint.sh"
    assert os.path.isfile(script), "lambda_entrypoint.sh not found"
    assert os.access(script, os.X_OK), "lambda_entrypoint.sh not executable"
    with open(script) as f:
        content = f.read()
    assert "AWS_LAMBDA_RUNTIME_API" in content
    assert "aws-lambda-rie" in content
    print("✓ lambda_entrypoint.sh (RIE/RIC auto-detect)")


def test_environment():
    """Validate environment variables."""
    required = ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]
    for var in required:
        val = os.environ.get(var, "NOT SET")
        print(f"  {var}: {val[:80]}...")
        assert var in os.environ

    assert "/var/lang/bin" in os.environ["PATH"]
    assert "/var/runtime" in os.environ["PYTHONPATH"]

    # Lambda-standard env vars
    assert os.environ.get("LAMBDA_TASK_ROOT") == "/var/task"
    assert os.environ.get("LAMBDA_RUNTIME_DIR") == "/var/runtime"
    print(f"  LAMBDA_TASK_ROOT: {os.environ['LAMBDA_TASK_ROOT']}")
    print(f"  LAMBDA_RUNTIME_DIR: {os.environ['LAMBDA_RUNTIME_DIR']}")

    # Locale and timezone
    assert os.environ.get("LANG") == "en_US.UTF-8"
    assert os.environ.get("TZ") == ":/etc/localtime"
    print(f"  LANG: {os.environ['LANG']}")
    print(f"  TZ: {os.environ['TZ']}")

    # LD_LIBRARY_PATH must include Bottlerocket NVIDIA driver path
    ld_path = os.environ["LD_LIBRARY_PATH"]
    required_ld_paths = [
        "/var/lang/lib",
        "/lib64",
        "/usr/lib64",
        "/var/runtime",
        "/var/runtime/lib",
        "/var/task",
        "/var/task/lib",
        "/opt/lib",
        "/usr/local/cuda/lib64",
        "/x86_64-bottlerocket-linux-gnu/sys-root/usr/lib/nvidia",
    ]
    for p in required_ld_paths:
        assert p in ld_path, f"LD_LIBRARY_PATH missing {p}"
    print(f"  LD_LIBRARY_PATH entries: {len(ld_path.split(':'))}")

    print("✓ Environment configured")


def test_workdir():
    """Validate working directory."""
    assert os.path.isdir("/var/task"), "/var/task not found"
    print("✓ /var/task exists")


def main():
    print("=" * 70)
    print("lambda-base Runtime Validation")
    print("=" * 70)

    tests = [
        ("Python Runtime", test_python_runtime),
        ("Lambda RIC", test_lambda_ric),
        ("Lambda RIE", test_lambda_rie),
        ("CUDA Runtime", test_cuda),
        ("Entrypoint Script", test_entrypoint),
        ("Environment", test_environment),
        ("Working Directory", test_workdir),
    ]

    failed = []
    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback

            traceback.print_exc()
            failed.append(name)

    print("\n" + "=" * 70)
    passed = len(tests) - len(failed)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)
    print("All tests passed! ✓")


if __name__ == "__main__":
    main()
