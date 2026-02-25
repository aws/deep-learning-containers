#!/usr/bin/env python3
"""ECR Enhanced Scan — poll for results and fail on non-allowlisted CRITICAL/HIGH CVEs.

Usage:
    python3 ecr_scan.py --image-uri <uri> --framework <name> \
        [--framework-version <ver>] [--allowlist-dir <dir>]

Allowlist resolution (merged in order):
  1. <allowlist-dir>/ecr_scan.json                            (global)
  2. <allowlist-dir>/<framework>/ecr_scan.json                (framework)
  3. <allowlist-dir>/<framework>/<framework>-<version>.json    (version-specific)

Each file: [{"vulnerability_id": "CVE-...", "reason": "..."}]
"""

import argparse
import json
import logging
import os
import sys
import time

from test_utils import ImageURI, parse_image_uri
from test_utils.aws import AWSSessionManager

import test  # noqa: F401 — triggers colored logging setup

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger("test").getChild("ecr_scan")
LOGGER.setLevel(logging.INFO)

SEVERITY_THRESHOLD = {"CRITICAL", "HIGH"}
POLL_INTERVAL = 30
POLL_TIMEOUT = 600  # 10 minutes


def wait_for_scan(ecr_client, image: ImageURI):
    """Poll until enhanced scan completes. Returns list of findings."""
    image_id = {"imageTag": image.image_tag}
    start = time.time()

    while time.time() - start < POLL_TIMEOUT:
        resp = ecr_client.describe_image_scan_findings(
            repositoryName=image.repository,
            imageId=image_id,
        )
        status = resp["imageScanStatus"]["status"]

        if status == "COMPLETE":
            findings = resp.get("imageScanFindings", {}).get("enhancedFindings", [])
            while resp.get("nextToken"):
                resp = ecr_client.describe_image_scan_findings(
                    repositoryName=image.repository,
                    imageId=image_id,
                    nextToken=resp["nextToken"],
                )
                findings.extend(resp.get("imageScanFindings", {}).get("enhancedFindings", []))
            return findings

        if status == "FAILED":
            desc = resp["imageScanStatus"].get("description", "unknown")
            LOGGER.error(f"Scan failed: {desc}")
            sys.exit(1)

        LOGGER.info(f"Scan status: {status}, waiting {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)

    LOGGER.error(f"Scan timed out after {POLL_TIMEOUT}s")
    sys.exit(1)


def load_allowlist(allowlist_dir, framework=None, framework_version=None):
    """Load and merge 3-level allowlist. Returns set of allowlisted vulnerability IDs."""
    paths = [os.path.join(allowlist_dir, "ecr_scan.json")]
    if framework:
        paths.append(os.path.join(allowlist_dir, framework, "ecr_scan.json"))
        if framework_version:
            paths.append(
                os.path.join(allowlist_dir, framework, f"{framework}-{framework_version}.json")
            )

    allowed = set()
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                for entry in json.load(f):
                    allowed.add(entry["vulnerability_id"])
    return allowed


def filter_findings(findings, allowlist):
    """Return CRITICAL/HIGH findings not covered by allowlist."""
    failures = []
    for vuln in findings:
        severity = vuln.get("severity", "")
        if severity not in SEVERITY_THRESHOLD:
            continue
        vuln_id = vuln.get("packageVulnerabilityDetails", {}).get("vulnerabilityId", "")
        if vuln_id in allowlist:
            continue
        for pkg in vuln.get("packageVulnerabilityDetails", {}).get("vulnerablePackages", [{}]):
            failures.append(
                {
                    "vulnerability_id": vuln_id,
                    "severity": severity,
                    "package": pkg.get("name", ""),
                    "installed_version": pkg.get("version", ""),
                    "fixed_in": pkg.get("fixedInVersion", "N/A"),
                    "source_url": vuln.get("packageVulnerabilityDetails", {}).get("sourceUrl", ""),
                    "description": vuln.get("description", ""),
                }
            )
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--framework-version", default="")
    parser.add_argument(
        "--allowlist-dir",
        default="test/security/data/ecr_scan_allowlist",
    )
    args = parser.parse_args()

    image = parse_image_uri(args.image_uri)
    ecr_client = AWSSessionManager(region=image.region).ecr

    LOGGER.info(f"Waiting for ECR enhanced scan: {image.repository}:{image.image_tag}")
    findings = wait_for_scan(ecr_client, image)
    LOGGER.info(f"Scan complete: {len(findings)} total findings")
    LOGGER.debug(f"All findings: {json.dumps(findings, indent=2, default=str)}")

    allowlist = load_allowlist(args.allowlist_dir, args.framework, args.framework_version)
    failures = filter_findings(findings, allowlist)

    if failures:
        LOGGER.error(f"{len(failures)} non-allowlisted CRITICAL/HIGH vulnerabilities:")
        for v in failures:
            LOGGER.error(
                f"  {v['severity']} {v['vulnerability_id']}\n"
                f"    Package: {v['package']} ({v['installed_version']} → {v['fixed_in']})\n"
                f"    URL: {v['source_url']}\n"
                f"    Description: {v['description'][:200]}"
            )
        return 1

    LOGGER.info("ECR scan passed (all CRITICAL/HIGH findings allowlisted or absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
