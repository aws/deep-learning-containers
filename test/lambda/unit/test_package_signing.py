"""Verify OS and CUDA packages are installed with signature verification enabled."""

import glob
import subprocess

import pytest

REPO_FILES = glob.glob("/etc/yum.repos.d/*.repo")


def _repo_text():
    text = ""
    for path in REPO_FILES:
        with open(path) as f:
            text += f.read()
    return text


class TestPackageSignatureVerification:
    """dnf must verify RPM signatures against pinned vendor keys."""

    def test_repo_files_present(self):
        assert REPO_FILES, "no dnf repo definitions found"

    def test_gpgcheck_enabled_everywhere(self):
        lines = [
            ln.strip() for ln in _repo_text().splitlines() if ln.strip().startswith("gpgcheck")
        ]
        assert lines, "no gpgcheck directive found in any repo"
        assert all(ln.endswith("1") for ln in lines), f"gpgcheck disabled somewhere: {lines}"

    @pytest.mark.parametrize("key", ["RPM-GPG-KEY-amazon-linux-2023", "RPM-GPG-KEY-NVIDIA"])
    def test_vendor_key_pinned_on_disk(self, key):
        assert glob.glob(f"/etc/pki/rpm-gpg/{key}*"), f"{key} not present"

    @pytest.mark.parametrize("owner", ["Amazon Linux", "cudatools"])
    def test_vendor_key_imported_into_rpmdb(self, owner):
        out = subprocess.run(
            ["rpm", "-q", "gpg-pubkey", "--qf", "%{SUMMARY}\n"],
            capture_output=True,
            text=True,
        ).stdout
        assert owner in out, f"{owner} key not imported; got: {out!r}"
