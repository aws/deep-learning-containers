#!/usr/bin/env python3
"""Upload per-image framework allowlists to S3.

Reads .github/config/image/*.yml to discover released images, looks up each
image's SHA in ECR, reads the framework's allowlist from the repo, and uploads
to s3://$SCANNER_ALLOWLIST_S3_BUCKET/<sha>/ecr_allowlist.json.

Usage:
    python3 .github/scripts/upload_ecr_allowlists.py              # all images
    python3 .github/scripts/upload_ecr_allowlists.py --dry-run    # print only
    python3 .github/scripts/upload_ecr_allowlists.py --image vllm:omni-cuda-v1
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import boto3
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / ".github" / "config" / "image"
ALLOWLIST_DIR = REPO_ROOT / "test" / "security" / "data" / "ecr_scan_allowlist"
ECR_ACCOUNT = os.environ.get("ECR_ACCOUNT_ID", "")
ECR_REGION = os.environ.get("AWS_REGION", "us-west-2")
S3_BUCKET = os.environ.get("SCANNER_ALLOWLIST_S3_BUCKET", "")


def load_image_configs():
    """Read all .github/config/image/*.yml and return list of (framework, repo, tag) tuples."""
    configs = []
    for path in sorted(CONFIG_DIR.glob("*.yml")):
        try:
            data = yaml.safe_load(path.read_text())
        except Exception as e:
            LOG.warning(f"skip {path.name}: can't parse ({e})")
            continue
        common = data.get("common", {})
        prod_image = common.get("prod_image")
        framework = common.get("framework")
        if not prod_image or not framework:
            LOG.debug(f"skip {path.name}: no prod_image or framework")
            continue
        if ":" not in prod_image:
            LOG.warning(f"skip {path.name}: prod_image '{prod_image}' has no tag")
            continue
        repo, tag = prod_image.split(":", 1)
        configs.append((framework, repo, tag, path.name))
    return configs


def get_image_sha(ecr_client, repo, tag):
    """Look up image digest from ECR. Returns sha string or None."""
    try:
        resp = ecr_client.describe_images(
            registryId=ECR_ACCOUNT,
            repositoryName=repo,
            imageIds=[{"imageTag": tag}],
        )
        details = resp.get("imageDetails", [])
        if details:
            return details[0]["imageDigest"]
    except ecr_client.exceptions.ImageNotFoundException:
        return None
    except Exception as e:
        LOG.warning(f"ecr lookup failed for {repo}:{tag}: {e}")
    return None


def load_framework_allowlist(framework):
    """Read the framework's allowlist from the repo. Returns list of entries or None."""
    path = ALLOWLIST_DIR / framework / "framework_allowlist.json"
    if not path.exists():
        return None
    try:
        entries = json.loads(path.read_text())
        if not isinstance(entries, list):
            LOG.warning(f"allowlist for {framework} is not a list")
            return None
        return entries
    except Exception as e:
        LOG.warning(f"can't read allowlist for {framework}: {e}")
        return None


def convert_to_scanner_format(entries):
    """Convert framework_allowlist.json format to what the scanner reads.

    Scanner reads: {"<package_key>": [{"vulnerability_id": ..., "reason_to_ignore": ...}, ...]}
    We use a single key "framework_allowlist" wrapping all entries.
    """
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict) or "vulnerability_id" not in entry:
            continue
        normalized.append(
            {
                "vulnerability_id": entry["vulnerability_id"],
                "reason_to_ignore": entry.get("reason", ""),
            }
        )
    return {"framework_allowlist": normalized}


def upload_to_s3(s3_client, sha, data, dry_run=False):
    """Upload allowlist JSON to s3://bucket/<sha>/ecr_allowlist.json."""
    key = f"{sha}/ecr_allowlist.json"
    body = json.dumps(data, indent=4).encode("utf-8")
    if dry_run:
        LOG.info(
            f"  [dry-run] would upload {len(data['framework_allowlist'])} entries to s3://{S3_BUCKET}/{key}"
        )
        return True
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=body)
        LOG.info(f"  uploaded {len(data['framework_allowlist'])} entries to s3://{S3_BUCKET}/{key}")
        return True
    except Exception as e:
        LOG.error(f"  s3 upload failed for {key}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be uploaded without writing to S3",
    )
    parser.add_argument("--image", help="target a single image (repo:tag), e.g. vllm:omni-cuda-v1")
    parser.add_argument("--region", default=ECR_REGION)
    args = parser.parse_args()

    if not ECR_ACCOUNT:
        LOG.error("ECR_ACCOUNT_ID env var not set")
        return 1
    if not S3_BUCKET:
        LOG.error("SCANNER_ALLOWLIST_S3_BUCKET env var not set")
        return 1

    configs = load_image_configs()
    LOG.info(f"found {len(configs)} image configs with prod_image")

    if args.image:
        target_repo, target_tag = args.image.split(":", 1)
        configs = [
            (fw, repo, tag, src)
            for fw, repo, tag, src in configs
            if repo == target_repo and tag == target_tag
        ]
        if not configs:
            LOG.error(f"no config found for --image {args.image}")
            return 1
        LOG.info(f"targeting single image: {args.image}")

    ecr_client = boto3.client("ecr", region_name=args.region)
    s3_client = boto3.client("s3", region_name=args.region)

    # Dedupe by (repo, tag) — multiple configs might point to same image
    seen = set()
    uploaded = 0
    skipped = 0
    failed = 0

    for framework, repo, tag, source_file in configs:
        image_key = f"{repo}:{tag}"
        if image_key in seen:
            continue
        seen.add(image_key)

        LOG.info(f"{image_key} (framework={framework}, config={source_file})")

        sha = get_image_sha(ecr_client, repo, tag)
        if not sha:
            LOG.warning("  skip: image not found in ECR")
            skipped += 1
            continue

        entries = load_framework_allowlist(framework)
        if entries is None:
            LOG.warning(f"  skip: no allowlist dir for framework '{framework}'")
            skipped += 1
            continue

        if not entries:
            LOG.info("  skip: allowlist is empty")
            skipped += 1
            continue

        data = convert_to_scanner_format(entries)
        if upload_to_s3(s3_client, sha, data, dry_run=args.dry_run):
            uploaded += 1
        else:
            failed += 1

    LOG.info(f"done: {uploaded} uploaded, {skipped} skipped, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
