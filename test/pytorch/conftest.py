"""PyTorch DLC test configuration.

Tests run directly inside the container — the workflow handles all Docker
orchestration (start container, volume-mount /workdir, attach GPUs, etc.).
No Docker-related fixtures are needed.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--config-file",
        default="",
        help="Path to image config YAML (e.g., '.github/config/image/pytorch/2.11-ec2-cpu.yml')",
    )
    parser.addoption(
        "--workdir",
        default="/workdir",
        help="Path where the DLC repo is mounted inside the container",
    )
