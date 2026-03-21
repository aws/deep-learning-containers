"""Verify SSH configuration for base and EKS images."""


def test_sshd_binary(container_exec):
    container_exec("test -x /usr/sbin/sshd")


def test_root_authorized_keys(container_exec):
    container_exec("test -f /root/.ssh/authorized_keys")


def test_strict_host_key_checking_disabled(container_exec):
    out = container_exec("cat /root/.ssh/config")
    assert "StrictHostKeyChecking no" in out
