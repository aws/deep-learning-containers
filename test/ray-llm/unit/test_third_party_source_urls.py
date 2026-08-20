"""Assert the ray-haproxy attribution matches the shipped ray-haproxy version."""

import re
import subprocess
from pathlib import Path


def test_haproxy_url_matches_shipped_version():
    attribution = Path("/root/THIRD_PARTY_SOURCE_CODE_URLS").read_text()
    m = re.search(r"ray-haproxy\s+(\S+)", attribution)
    assert m, "THIRD_PARTY_SOURCE_CODE_URLS missing ray-haproxy entry"
    attributed = m.group(1)

    result = subprocess.run(
        ["pip", "show", "ray-haproxy"], capture_output=True, text=True, check=True
    )
    shipped = re.search(r"^Version:\s+(\S+)", result.stdout, re.MULTILINE).group(1)

    assert attributed == shipped, (
        f"ray-haproxy version drift: attribution {attributed!r} vs shipped {shipped!r}. "
        f"Update THIRD_PARTY_SOURCE_CODE_URLS and re-upload the source tarball."
    )
