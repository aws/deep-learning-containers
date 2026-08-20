"""Unit tests for telemetry OS detection.

These run locally against real /etc/os-release payloads — no EC2 instance or
image is required, so they execute alongside the instance telemetry tests in
_reusable.telemetry-tests.yml.
"""

import pytest
from deep_learning_container import _retrieve_os

# Verbatim /etc/os-release excerpts from the base images DLC builds on.
AMZN2023 = """\
NAME="Amazon Linux"
VERSION="2023"
ID="amzn"
ID_LIKE="fedora"
VERSION_ID="2023"
PLATFORM_ID="platform:al2023"
PRETTY_NAME="Amazon Linux 2023.12.20260727"
CPE_NAME="cpe:2.3:o:amazon:amazon_linux:2023"
HOME_URL="https://aws.amazon.com/linux/amazon-linux-2023/"
SUPPORT_END="2029-06-30"
"""

UBUNTU_24_04 = """\
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
UBUNTU_CODENAME=noble
"""

UBUNTU_22_04 = """\
PRETTY_NAME="Ubuntu 22.04.5 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.5 LTS (Jammy Jellyfish)"
ID=ubuntu
ID_LIKE=debian
UBUNTU_CODENAME=jammy
"""

UBUNTU_20_04 = """\
NAME="Ubuntu"
VERSION="20.04.6 LTS (Focal Fossa)"
ID=ubuntu
ID_LIKE=debian
VERSION_ID="20.04"
UBUNTU_CODENAME=focal
"""


@pytest.mark.parametrize(
    "content, expected",
    [
        # Amazon Linux 2023 quotes ID and ships an undotted VERSION_ID.
        (AMZN2023, "amzn2023"),
        (UBUNTU_24_04, "ubuntu24.04"),
        (UBUNTU_22_04, "ubuntu22.04"),
        (UBUNTU_20_04, "ubuntu20.04"),
    ],
    ids=["amzn2023", "ubuntu24.04", "ubuntu22.04", "ubuntu20.04"],
)
def test_retrieve_os_matches_image_os_version(tmp_path, content, expected):
    """OS string matches the os_version naming used in .github/config/image."""
    path = tmp_path / "os-release"
    path.write_text(content)
    assert _retrieve_os(str(path)) == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        ('ID=ubuntu\nVERSION_ID="24.04"\n', "ubuntu24.04"),
        ('ID="amzn"\nVERSION_ID="2023"\n', "amzn2023"),
        ("ID='amzn'\nVERSION_ID='2023'\n", "amzn2023"),
        ("ID=amzn\nVERSION_ID=2023\n", "amzn2023"),
    ],
    ids=["bare-id", "double-quoted", "single-quoted", "all-bare"],
)
def test_retrieve_os_handles_quoting_variants(tmp_path, content, expected):
    path = tmp_path / "os-release"
    path.write_text(content)
    assert _retrieve_os(str(path)) == expected


def test_retrieve_os_ignores_keys_that_merely_start_with_id(tmp_path):
    """ID_LIKE / VERSION_ID must not be mistaken for ID."""
    path = tmp_path / "os-release"
    path.write_text('ID_LIKE="fedora"\nID="amzn"\nVERSION_ID="2023"\n')
    assert _retrieve_os(str(path)) == "amzn2023"


def test_retrieve_os_missing_file_does_not_raise(tmp_path):
    """Telemetry runs at shell startup in customer containers; it must not crash."""
    assert _retrieve_os(str(tmp_path / "does-not-exist")) == ""
