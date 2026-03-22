"""Verify SSH configuration for base and EKS images."""

import os


def test_sshd_binary():
    assert os.access("/usr/sbin/sshd", os.X_OK)


def test_root_authorized_keys():
    assert os.path.isfile("/root/.ssh/authorized_keys")


def test_strict_host_key_checking_disabled():
    with open("/root/.ssh/config") as f:
        assert "StrictHostKeyChecking no" in f.read()


def test_sshd_port_22():
    """Port 22 must be the effective port (sshd default, not overridden)."""
    with open("/etc/ssh/sshd_config") as f:
        content = f.read()
    # Ensure EKS override hasn't leaked into the base image
    assert "Port 2222" not in content


def test_sshd_root_login_not_disabled():
    with open("/etc/ssh/sshd_config") as f:
        content = f.read()
    # sshd default is PermitRootLogin prohibit-password; ensure it's not "no"
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("PermitRootLogin"):
            assert "no" not in stripped.lower().split()[-1], "Root login must not be disabled"
