"""Unit tests for scripts/docker/whisperx/start_cuda_compat.sh.

CPU-only, no container and no GPU: we copy the shipped script, retarget its
hard-coded compat-lib probe at a temp fixture, and simulate driver detection with
a fake ``nvidia-smi`` on PATH. This drives the script's driver-detection +
compat-activation branches directly, including the regression where an empty
``NVIDIA_DRIVER_VERSION`` (CPU host, no driver) must NOT abort the entrypoint: the
whisperx entrypoints ``source`` this under ``set -euo pipefail``, and the pre-fix
code passed the empty version unquoted to ``verlte`` -- collapsing two args to one
so ``verlte`` dereferenced an unbound ``$2`` and ``set -u`` killed the container
before uvicorn started. The ``[ -z ]`` guard + quoted args fix that.

The script reads ``/proc/driver/nvidia/version`` before falling back to
``nvidia-smi``; these tests assume a CPU host where that file is absent (true on
the CPU-only ``default-runner`` the unit suite runs on), so ``nvidia-smi`` is the
sole detection path we control.
"""

import os
import stat
import subprocess
from pathlib import Path

# start_cuda_compat.sh lives in the image-build tree, two levels up from test/whisperx/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docker" / "whisperx" / "start_cuda_compat.sh"

# The versioned lib the compat symlink points at. The script derives the max
# supported driver version by `readlink libcuda.so.1 | cut -d. -f3-`, so this name
# yields CUDA_COMPAT_MAX_DRIVER_VERSION=570.211.01 -- the pivot the tests bracket.
_COMPAT_SO = "libcuda.so.570.211.01"

# The literal probe path in the shipped script; we retarget only this at our
# fixture. The `export LD_LIBRARY_PATH=/usr/local/cuda/compat:...` line is left
# untouched so the "add" case can assert on that literal directory.
_PROBE_PATH = "/usr/local/cuda/compat/libcuda.so.1"


def _ldlp_line(out: str) -> str:
    """Return the wrapper's ``LDLP=[...]`` line ('' if the script aborted first)."""
    for line in out.splitlines():
        if line.startswith("LDLP="):
            return line
    return ""


def _run(tmp_path, *, compat_present: bool, driver_version: str | None):
    """Source a retargeted copy of the script and return the CompletedProcess.

    ``compat_present`` builds a fake compat dir (empty versioned lib + the
    ``libcuda.so.1`` symlink the script readlinks); otherwise the probe path points
    at a nonexistent file so the script's ``[ -f ]`` test fails. ``driver_version``
    drives a fake ``nvidia-smi`` on PATH -- ``None`` installs none, so detection
    yields an empty version (the CPU-host case).
    """
    if compat_present:
        compat_dir = tmp_path / "compat"
        compat_dir.mkdir()
        (compat_dir / _COMPAT_SO).write_bytes(b"")
        compat_lib = compat_dir / "libcuda.so.1"
        # Relative target so `readlink | cut -d. -f3-` yields "570.211.01".
        os.symlink(_COMPAT_SO, compat_lib)
    else:
        # Nonexistent path -> the script's `[ -f ]` probe fails ("package not found").
        compat_lib = tmp_path / "compat" / "libcuda.so.1"

    script_copy = tmp_path / "start_cuda_compat.sh"
    script_copy.write_text(SCRIPT.read_text().replace(_PROBE_PATH, str(compat_lib)))

    # Always stub nvidia-smi (bin_dir is first on PATH) so a host's real one can't
    # leak in; driver_version=None => an empty-output stub (the "no driver" case).
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    if driver_version is not None:
        nvidia_smi.write_text(f'#!/bin/sh\necho "{driver_version}"\n')
    else:
        nvidia_smi.write_text("#!/bin/sh\nexit 1\n")
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Source the copy exactly as the entrypoints do: under `set -euo pipefail`, with
    # LD_LIBRARY_PATH pre-seeded. We deliberately omit LD_LIBRARY_PATH from env so the
    # `${LD_LIBRARY_PATH:-}` preseed is what keeps the unguarded `$LD_LIBRARY_PATH`
    # reference in the script from tripping `set -u`.
    prog = (
        "set -euo pipefail; "
        'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"; '
        f'. "{script_copy}"; '
        'echo "REACHED_END"; '
        'echo "LDLP=[$LD_LIBRARY_PATH]"'
    )
    return subprocess.run(
        ["bash", "-c", prog],
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


def test_no_driver_does_not_abort_and_skips(tmp_path):
    """Regression: an empty driver version must not abort the sourced script.

    Pre-fix, the empty ``NVIDIA_DRIVER_VERSION`` reached ``verlte`` unquoted and
    ``set -u`` aborted on an unbound ``$2`` before uvicorn. Now the ``[ -z ]`` guard
    skips compat and leaves LD_LIBRARY_PATH empty.
    """
    cp = _run(tmp_path, compat_present=True, driver_version=None)
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "REACHED_END" in out
    assert "no NVIDIA driver was detected" in out
    assert "unbound variable" not in out
    assert _ldlp_line(out) == "LDLP=[]"  # compat dir NOT prepended


def test_old_driver_adds_compat(tmp_path):
    """An older driver (< compat max 570.211.01) prepends the compat dir."""
    cp = _run(tmp_path, compat_present=True, driver_version="400.00.00")
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "Adding CUDA compat to LD_LIBRARY_PATH" in out
    assert "/usr/local/cuda/compat" in _ldlp_line(out)


def test_new_driver_skips_compat(tmp_path):
    """A newer driver (> compat max) needs no compat, so it is left off LD_LIBRARY_PATH."""
    cp = _run(tmp_path, compat_present=True, driver_version="999.99.99")
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "newer NVIDIA driver is installed" in out
    assert "/usr/local/cuda/compat" not in _ldlp_line(out)


def test_compat_absent_skips(tmp_path):
    """No compat package (CPU image) -> the `[ -f ]` probe fails, script skips cleanly."""
    cp = _run(tmp_path, compat_present=False, driver_version=None)
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "package not found" in out


def test_failing_nvidia_smi_does_not_abort(tmp_path):
    """A broken nvidia-smi (present but no GPU) prints a multi-word error string as
    the "version"; quoting the verlte args must keep that from tripping ``set -u``,
    and compat is still skipped. Mirrors CPU hosts that ship nvidia-smi without a
    working driver (e.g. the CI CPU runner).
    """
    cp = _run(
        tmp_path,
        compat_present=True,
        driver_version="NVIDIA-SMI has failed because it could not communicate with the driver",
    )
    out = cp.stdout + cp.stderr
    assert cp.returncode == 0, out
    assert "REACHED_END" in out
    assert "unbound variable" not in out
    # A non-version string is not < the compat max, so compat is not prepended.
    assert "/usr/local/cuda/compat" not in _ldlp_line(out), out
