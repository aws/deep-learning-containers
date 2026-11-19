#!/usr/bin/env python3
"""Expose the pip-provided NVIDIA libraries through one linker directory."""

from __future__ import annotations

import glob
import os

TARGET = "/usr/local/nvidia/pip-lib"


def main() -> None:
    os.makedirs(TARGET, exist_ok=True)
    for directory in glob.glob("/opt/venv/lib/python*/site-packages/nvidia/*/lib"):
        for library in glob.glob(f"{directory}/*.so*"):
            link = os.path.join(TARGET, os.path.basename(library))
            if not os.path.exists(link):
                os.symlink(library, link)


if __name__ == "__main__":
    main()
