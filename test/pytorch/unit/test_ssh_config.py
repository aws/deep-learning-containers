"""Verify SSH configuration for base and EKS images."""


def test_sshd_binary(run_in_container):
    run_in_container("test -x /usr/sbin/sshd")


def test_root_authorized_keys(run_in_container):
    run_in_container("test -f /root/.ssh/authorized_keys")


def test_strict_host_key_checking_disabled(run_in_container):
    out = run_in_container("cat /root/.ssh/config")
    assert "StrictHostKeyChecking no" in out
