#!/usr/bin/env python3
"""Fail when pip metadata deviations differ from the reviewed allowlist."""

from __future__ import annotations

import pathlib
import subprocess
import sys


def _lines(text: str) -> set[str]:
    return {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    allowlist_path = pathlib.Path(sys.argv[1])
    expected = _lines(allowlist_path.read_text())
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        text=True,
        capture_output=True,
    )
    actual = _lines(result.stdout)

    unexpected = sorted(actual - expected)
    resolved_or_changed = sorted(expected - actual)
    if unexpected or resolved_or_changed:
        if unexpected:
            print("Unexpected pip-check findings:", file=sys.stderr)
            print("\n".join(f"  + {line}" for line in unexpected), file=sys.stderr)
        if resolved_or_changed:
            print("Allowlisted findings no longer present:", file=sys.stderr)
            print(
                "\n".join(f"  - {line}" for line in resolved_or_changed),
                file=sys.stderr,
            )
        raise SystemExit(1)

    print(f"pip-check deviations match reviewed allowlist ({len(actual)} findings)")


if __name__ == "__main__":
    main()
