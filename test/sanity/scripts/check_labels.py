#!/usr/bin/env python3
"""Check that all URI labels in a Docker image use https:// (public).

Usage: python3 check_labels.py <image-uri>
"""

import json
import subprocess
import sys


def main():
    image_uri = sys.argv[1]

    result = subprocess.run(
        ["docker", "inspect", "--format={{json .Config.Labels}}", image_uri],
        capture_output=True,
        text=True,
        check=True,
    )
    labels = json.loads(result.stdout.strip())

    if not labels:
        print("WARNING: No labels found on image")
        return 0

    failed = []
    for name, value in labels.items():
        if "uri" in name.lower() and not value.startswith("https://"):
            failed.append(f"  {name}: {value}")
        if len(name) > 128:
            failed.append(f"  Label name exceeds 128 chars: {name}")
        if len(value) > 256:
            failed.append(f"  Label value exceeds 256 chars for {name}: {value[:50]}...")

    if failed:
        print("FAIL: Label validation errors:")
        print("\n".join(failed))
        return 1

    print(f"All {len(labels)} labels pass validation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
