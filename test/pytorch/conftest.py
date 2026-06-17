"""PyTorch DLC test configuration.

Tests run directly inside the container — the workflow handles all Docker
orchestration (start container, volume-mount /workdir, attach GPUs, etc.).
No Docker-related fixtures are needed.
"""
