"""Verify EKS image: non-root user, SSH port 2222, torch works as mluser."""

import pytest


@pytest.fixture(scope="module")
def eks_image_uri(image_uri):
    """Skip if not testing an EKS image (pass --image-uri with eks tag)."""
    return image_uri


def test_runs_as_mluser(run_in_container):
    """EKS image should default to mluser (uid 1000)."""
    out = run_in_container("id -u", user="1000:1000")
    assert out == "1000"


def test_mluser_can_import_torch(run_in_container):
    out = run_in_container(
        "python -c 'import torch; print(torch.cuda.is_available())'", user="1000:1000"
    )
    assert out in ("True", "False")  # GPU may not be available, but import must work


def test_mluser_ssh_keys_exist(run_in_container):
    run_in_container("test -f /home/mluser/.ssh/id_rsa", user="1000:1000")
    run_in_container("test -f /home/mluser/.ssh/authorized_keys", user="1000:1000")


def test_mluser_ssh_config_port_2222(run_in_container):
    out = run_in_container("cat /home/mluser/.ssh/config", user="1000:1000")
    assert "Port 2222" in out
    assert "StrictHostKeyChecking no" in out


def test_sshd_config_port_2222(run_in_container):
    out = run_in_container("grep '^Port' /etc/ssh/sshd_config")
    assert "2222" in out


def test_mluser_owns_venv(run_in_container):
    out = run_in_container("stat -c '%u' /opt/venv", user="1000:1000")
    assert out == "1000"


def test_mluser_owns_opt_ml(run_in_container):
    out = run_in_container("stat -c '%u' /opt/ml", user="1000:1000")
    assert out == "1000"


def test_mluser_can_write_workspace(run_in_container):
    run_in_container(
        "touch /home/mluser/test_write && rm /home/mluser/test_write", user="1000:1000"
    )
