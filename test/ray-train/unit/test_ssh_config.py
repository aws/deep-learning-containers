"""Verify SSH configuration (worker comms; port 22)."""

import os


def test_sshd_binary():
    assert os.access("/usr/sbin/sshd", os.X_OK)


def test_root_authorized_keys():
    assert os.path.isfile("/root/.ssh/authorized_keys")


def test_strict_host_key_checking_disabled():
    with open("/root/.ssh/config") as f:
        assert "StrictHostKeyChecking no" in f.read()


def test_sshd_port_22():
    with open("/etc/ssh/sshd_config") as f:
        assert "Port 2222" not in f.read()
