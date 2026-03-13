#!/usr/bin/env python3
"""Clone third-party source repos and upload tarballs to S3 for OSS compliance.

Reads THIRD_PARTY_SOURCE_CODE_URLS (format: <name> <version> <url> per line),
clones each repo, tars it, and uploads to S3 if not already present.

Usage: python3 upload_oss_source.py <urls-file>
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time

import botocore
from test_utils.aws import AWSSessionManager

import test  # noqa: F401 — triggers colored logging setup

# To enable debugging, change logging.INFO to logging.DEBUG
LOGGER = logging.getLogger("test").getChild("upload_oss_source")
LOGGER.setLevel(logging.INFO)

BUCKET = "aws-dlinfra-licenses"
BUCKET_PATH = "third_party_source_code"
MAX_RETRIES = 3


def already_on_s3(s3_client, s3_key):
    """Check if object already exists on S3."""
    try:
        s3_client.head_object(Bucket=BUCKET, Key=s3_key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def upload_to_s3(s3_client, local_path, s3_key):
    """Upload file to S3 and set public-read ACL."""
    LOGGER.info(f"Uploading to s3://{BUCKET}/{s3_key}")
    s3_client.upload_file(local_path, BUCKET, s3_key)
    s3_client.put_object_acl(Bucket=BUCKET, Key=s3_key, ACL="public-read")


def clone_and_tar(name, version, url, work_dir):
    """Clone repo or download tarball and create tarball. Returns tarball path or None."""
    dir_name = f"{name}_v{version}_source_code"
    local_dir = os.path.join(work_dir, dir_name)
    tarball = f"{local_dir}.tar.gz"
    url = url.strip()
    is_tarball = url.endswith((".tar.gz", ".tar.xz", ".tar.bz2", ".tgz"))

    for attempt in range(MAX_RETRIES):
        try:
            if not os.path.isdir(local_dir):
                if is_tarball:
                    downloaded = os.path.join(work_dir, os.path.basename(url))
                    subprocess.run(
                        ["curl", "-fsSL", "-o", downloaded, url], check=True, capture_output=True
                    )
                    os.makedirs(local_dir)
                    subprocess.run(
                        ["tar", "-xf", downloaded, "-C", local_dir, "--strip-components=1"],
                        check=True,
                        capture_output=True,
                    )
                else:
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "--branch", version, url, local_dir],
                        check=True,
                        capture_output=True,
                    )
            subprocess.run(
                ["tar", "-czf", tarball, "-C", work_dir, dir_name], check=True, capture_output=True
            )
            return tarball
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                LOGGER.error(f"Failed to clone {url} after {MAX_RETRIES} attempts: {e}")
                return None
            time.sleep(1)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Clone third-party source repos and upload tarballs to S3"
    )
    parser.add_argument("urls_file", help="Path to THIRD_PARTY_SOURCE_CODE_URLS file")
    args = parser.parse_args()

    if not os.path.exists(args.urls_file):
        LOGGER.error(f"{args.urls_file} not found")
        return 1

    s3_client = AWSSessionManager().s3
    failed = []

    with tempfile.TemporaryDirectory() as work_dir:
        with open(args.urls_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    LOGGER.warning(f"Malformed line: {line}")
                    continue
                name, version, url = parts[0], parts[1], parts[2]
                s3_key = f"{BUCKET_PATH}/{name}_v{version}_source_code.tar.gz"

                LOGGER.info(f"Processing: {name} v{version}")
                if already_on_s3(s3_client, s3_key):
                    LOGGER.info(f"Already exists on S3: {s3_key}, skipping")
                    continue
                tarball = clone_and_tar(name, version, url, work_dir)
                if tarball:
                    try:
                        upload_to_s3(s3_client, tarball, s3_key)
                    except Exception as e:
                        LOGGER.error(f"Failed to upload {name}: {e}")
                        failed.append(name)
                else:
                    failed.append(name)

    if failed:
        LOGGER.error(f"Could not process: {', '.join(failed)}")
        return 1

    LOGGER.info("All third-party source code uploaded successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
