#!/usr/bin/env python3
"""Verify the TransformerEngine wheel triplet installed for Miles.

The version is pinned by the image lock; this checks that the three dists agree
with each other and that transformer_engine_torch's runtime deps are installed.
"""

from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import re
import sys


def _dist_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def verify(core_dist: str) -> None:
    core = core_dist.replace("_", "-")
    version = metadata.version("transformer-engine")
    expected = {
        "transformer-engine": version,
        core: version,
        "transformer-engine-torch": version,
    }
    actual = {name: metadata.version(name) for name in expected}
    raw_requires = metadata.requires("transformer-engine-torch") or []
    requires = {
        requirement.lower().replace("_", "-").replace(" ", "") for requirement in raw_requires
    }

    assert actual == expected, f"unexpected TransformerEngine versions: {actual}"
    assert f"{core}=={version}" in requires, (
        f"unexpected TransformerEngine torch requirements: {requires}"
    )
    assert importlib.util.find_spec("transformer_engine") is not None, (
        "transformer_engine package not found"
    )

    missing = []
    for requirement in raw_requires:
        name = _dist_name(requirement)
        if name.startswith("transformer-engine"):
            continue
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    assert not missing, f"transformer_engine_torch runtime deps not installed: {missing}"
    print(f"TransformerEngine {version} verified ({core})")


if __name__ == "__main__":
    verify(sys.argv[1])
