"""Verify pytorch-image-specific environment — pytorch image."""

import os


def test_ld_library_path_includes_usr_local_lib():
    """/usr/local/lib must be in LD_LIBRARY_PATH for FFmpeg shared libs."""
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    assert "/usr/local/lib" in ld


def test_nvidia_driver_capabilities():
    assert os.environ.get("NVIDIA_DRIVER_CAPABILITIES") == "compute,utility,video"
