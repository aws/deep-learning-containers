"""PyTorch DLC test configuration.

Tests run directly inside the container — the workflow handles all Docker
orchestration (start container, volume-mount /workdir, attach GPUs, etc.).
No Docker-related fixtures are needed.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--pytorch-version",
        required=True,
        help="Short PyTorch version for locating version pins (e.g., '2.11')",
    )
    parser.addoption(
        "--workdir",
        default="/workdir",
        help="Path where the DLC repo is mounted inside the container",
    )
