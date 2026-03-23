"""Verify EKS image: non-root user, SSH port 2222, torch works as mluser."""

import os
import pwd


def test_runs_as_mluser():
    """EKS image should default to mluser (uid 1000)."""
    assert os.getuid() == 1000


def test_mluser_can_import_torch():
    import torch

    assert hasattr(torch, "__version__")


def test_mluser_ssh_keys_exist():
    home = pwd.getpwuid(1000).pw_dir
    assert os.path.isfile(f"{home}/.ssh/id_rsa")
    assert os.path.isfile(f"{home}/.ssh/authorized_keys")


def test_mluser_ssh_config_port_2222():
    home = pwd.getpwuid(1000).pw_dir
    with open(f"{home}/.ssh/config") as f:
        content = f.read()
    assert "Port 2222" in content
    assert "StrictHostKeyChecking no" in content


def test_sshd_config_port_2222():
    with open("/etc/ssh/sshd_config") as f:
        content = f.read()
    assert any("Port" in line and "2222" in line for line in content.splitlines())


def test_mluser_owns_venv():
    assert os.stat("/opt/venv").st_uid == 1000


def test_mluser_owns_opt_ml():
    assert os.stat("/opt/ml").st_uid == 1000


def test_mluser_can_write_workspace():
    home = pwd.getpwuid(1000).pw_dir
    test_file = f"{home}/test_write"
    with open(test_file, "w") as f:
        f.write("ok")
    os.remove(test_file)
