#!/usr/bin/env python3
"""ECR Enhanced Scan — poll for results and fail on non-allowlisted CRITICAL/HIGH CVEs.

Usage:
    python3 ecr_scan.py --image-uri <uri> --framework <name> \
        [--framework-version <ver>] [--allowlist-dir <dir>]

Allowlist resolution (merged in order):
  1. <allowlist-dir>/global_allowlist.json                            (global)
  2. <allowlist-dir>/<framework>/framework_allowlist.json              (framework)
  3. <allowlist-dir>/<framework>/<framework>-<version>.json            (version-specific)

Each file: [{"vulnerability_id": "CVE-...", "reason": "..."}]
"""

import argparse
import json
import logging
import os
import sys
import time
from pprint import pformat

from test_utils import wait_for_status
from test_utils.aws import AWSSessionManager
from test_utils.docker_helper import ImageURI, parse_image_uri

import test  # noqa: F401 — triggers colored logging setup

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger("test").getChild("ecr_scan")
LOGGER.setLevel(logging.INFO)

SEVERITY_THRESHOLD = {"CRITICAL", "HIGH"}
SCAN_WAIT_PERIOD = 20
SCAN_WAIT_LENGTH = 30
SCAN_COMPLETE = "COMPLETE"
SCAN_POST_COMPLETE_WAIT = 120  # additional wait after scan completes before reading findings
GLOBAL_ALLOWLIST_FILE = "global_allowlist.json"
FRAMEWORK_ALLOWLIST_FILE = "framework_allowlist.json"


def get_scan_status(ecr_client, repository: str, image_tag: str) -> str:
    resp = ecr_client.describe_image_scan_findings(
        repositoryName=repository,
        imageId={"imageTag": image_tag},
    )
    return resp["imageScanStatus"]["status"]


def get_scan_findings(ecr_client, image: ImageURI) -> list:
    """Retrieve all paginated enhanced scan findings."""
    image_id = {"imageTag": image.image_tag}
    resp = ecr_client.describe_image_scan_findings(
        repositoryName=image.repository,
        imageId=image_id,
        maxResults=100,
    )
    findings = resp.get("imageScanFindings", {}).get("enhancedFindings", [])
    while resp.get("nextToken"):
        resp = ecr_client.describe_image_scan_findings(
            repositoryName=image.repository,
            imageId=image_id,
            maxResults=100,
            nextToken=resp["nextToken"],
        )
        findings.extend(resp.get("imageScanFindings", {}).get("enhancedFindings", []))
    return findings


def load_allowlist(allowlist_dir, framework=None, framework_version=None):
    """Load and merge 3-level allowlist. Returns set of allowlisted vulnerability IDs.

    Fails if any entry has a 'review_by' date that has passed.
    """
    import datetime

    paths = [os.path.join(allowlist_dir, GLOBAL_ALLOWLIST_FILE)]
    if framework:
        paths.append(os.path.join(allowlist_dir, framework, FRAMEWORK_ALLOWLIST_FILE))
        if framework_version:
            paths.append(
                os.path.join(allowlist_dir, framework, f"{framework}-{framework_version}.json")
            )

    today = datetime.date.today()
    expired = []
    allowed = set()
    for path in paths:
        if os.path.exists(path):
            with open(path) as f:
                for entry in json.load(f):
                    allowed.add(entry["vulnerability_id"])
                    review_by = entry.get("review_by")
                    if review_by:
                        due = datetime.date.fromisoformat(review_by)
                        if due < today:
                            expired.append(
                                f"{entry['vulnerability_id']} (review_by {review_by}, {path})"
                            )

    if expired:
        LOGGER.error(
            f"{len(expired)} allowlist entries past their review_by date — "
            "update or remove them:\n" + "\n".join(f"  {e}" for e in expired)
        )
        sys.exit(1)

    return allowed


def is_esm_fix(fixed_version: str) -> bool:
    """Ubuntu Pro/ESM fixes require a paid subscription — skip these."""
    v = fixed_version.lower()
    return "esm" in v and "ubuntu" in v


def filter_findings(findings, allowlist):
    """Return CRITICAL/HIGH findings not covered by allowlist, grouped by CVE."""
    grouped = {}
    for vuln in findings:
        if vuln.get("status") != "ACTIVE":
            continue
        severity = vuln.get("severity", "")
        if severity not in SEVERITY_THRESHOLD:
            continue
        vuln_id = vuln.get("packageVulnerabilityDetails", {}).get("vulnerabilityId", "")
        if vuln_id in allowlist:
            continue

        packages = []
        seen_pkgs = set()
        for pkg in vuln.get("packageVulnerabilityDetails", {}).get("vulnerablePackages", [{}]):
            fixed_in = pkg.get("fixedInVersion", "N/A")
            if fixed_in in ("N/A", "NotAvailable"):
                continue
            if is_esm_fix(fixed_in):
                continue
            pkg_key = (pkg.get("name", ""), pkg.get("version", ""), fixed_in)
            if pkg_key in seen_pkgs:
                continue
            seen_pkgs.add(pkg_key)
            packages.append(
                {
                    "name": pkg.get("name", ""),
                    "version": pkg.get("version", ""),
                    "fixed_in": fixed_in,
                    "file_path": pkg.get("filePath", ""),
                }
            )

        if not packages:
            continue

        if vuln_id not in grouped:
            grouped[vuln_id] = {
                "vulnerability_id": vuln_id,
                "severity": severity,
                "source_url": vuln.get("packageVulnerabilityDetails", {}).get("sourceUrl", ""),
                "description": vuln.get("description", ""),
                "manager": vuln.get("packageVulnerabilityDetails", {})
                .get("vulnerablePackages", [{}])[0]
                .get("packageManager", ""),
                "packages": [],
            }
        grouped[vuln_id]["packages"].extend(packages)
    return list(grouped.values())


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
    img_resp = ecr_client.describe_images(
        repositoryName=image.repository,
        imageIds=[{"imageTag": image.image_tag}],
    )
    sha = img_resp["imageDetails"][0]["imageDigest"]

    LOGGER.info(f"Waiting for ECR enhanced scan: {image.repository}:{image.image_tag} ({sha})")
    assert wait_for_status(
        SCAN_COMPLETE,
        SCAN_WAIT_PERIOD,
        SCAN_WAIT_LENGTH,
        get_scan_status,
        ecr_client,
        image.repository,
        image.image_tag,
    )

    LOGGER.info(f"Waiting {SCAN_POST_COMPLETE_WAIT}s for findings to stabilize...")
    time.sleep(SCAN_POST_COMPLETE_WAIT)

    findings = get_scan_findings(ecr_client, image)
    LOGGER.info(f"Scan complete: {len(findings)} findings across all severities")
    LOGGER.debug(f"All findings: {json.dumps(findings, indent=2, default=str)}")

    allowlist = load_allowlist(args.allowlist_dir, args.framework, args.framework_version)
    failures = filter_findings(findings, allowlist)

    if failures:
        LOGGER.error(f"{len(failures)} non-allowlisted CRITICAL/HIGH CVEs:")
        for vuln in failures:
            pkg_summary = ", ".join(
                f"{pkg['name']} ({pkg['version']} → {pkg['fixed_in']})" for pkg in vuln["packages"]
            )
            pin_suggestions = ", ".join(
                f"{pkg['name']}>={pkg['fixed_in']}"
                for pkg in vuln["packages"]
                if pkg["fixed_in"] != "N/A"
            )
            file_paths = ", ".join(
                pkg["file_path"] for pkg in vuln["packages"] if pkg.get("file_path")
            )
            allowlist_entry = pformat(
                {"vulnerability_id": vuln["vulnerability_id"], "reason": "TODO"}
            )
            LOGGER.error(
                f"{vuln['severity']} {vuln['vulnerability_id']}\n"
                f"\tPackage Manager: {vuln['manager']}\n"
                f"\tPackages: {pkg_summary}\n"
                f"\tFile paths: {file_paths}\n"
                f"\tURL: {vuln['source_url']}\n"
                f"\tDescription: {vuln['description'][:200]}\n"
                f"\tPin fix: {pin_suggestions}\n"
                f"\tAllowlist entry: {allowlist_entry}"
            )
        return 1

    LOGGER.info("ECR scan passed (all CRITICAL/HIGH findings allowlisted or absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
