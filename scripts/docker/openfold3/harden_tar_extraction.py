#!/usr/bin/env python3
"""Add filter="data" to the upstream ColabFold MSA client's tarfile.extractall() calls.

The MSA server response is unpacked with no member-path validation, so a tar entry whose
path escapes the target directory writes anywhere the process can (CVE-2007-4559). Python
3.12's filter="data" rejects absolute paths, parent traversal, outbound links, and special
files. Applied at build time and asserted, so a version bump that moves these call sites
fails the build instead of silently dropping the fix. Remove once upstream carries it
(absent as of 0.5.1).
"""

import sys
from pathlib import Path

REPLACEMENTS = (
    ("tar_gz.extractall(path)", 'tar_gz.extractall(path, filter="data")'),
    ("tar.extractall(path=TMPL_PATH)", 'tar.extractall(path=TMPL_PATH, filter="data")'),
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <colabfold_msa_server.py>", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    source = target.read_text()

    for old, new in REPLACEMENTS:
        if new in source:
            continue
        if source.count(old) != 1:
            print(
                f"expected one occurrence of {old!r} in {target}, found {source.count(old)}",
                file=sys.stderr,
            )
            return 1
        source = source.replace(old, new)

    target.write_text(source)

    remaining = source.replace('extractall(path, filter="data")', "").replace(
        'extractall(path=TMPL_PATH, filter="data")', ""
    )
    if "extractall" in remaining:
        print(f"unfiltered extractall() still present in {target}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
