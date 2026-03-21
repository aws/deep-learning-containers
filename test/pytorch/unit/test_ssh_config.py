"""Verify SSH configuration for base and EKS images."""

import os


def test_sshd_binary():
    assert os.access("/usr/sbin/sshd", os.X_OK)


def test_root_authorized_keys():
    assert os.path.isfile("/root/.ssh/authorized_keys")


def test_strict_host_key_checking_disabled():
    with open("/root/.ssh/config") as f:
        assert "StrictHostKeyChecking no" in f.read()
