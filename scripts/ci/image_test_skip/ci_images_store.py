#!/usr/bin/env python3
"""dlc-ci-images test-skip cache: read (check skip) and write (record pass)."""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

TABLE_NAME = "dlc-ci-images"
TABLE_REGION = "us-west-2"
TEST_ROW_TTL_SECONDS = 7 * 24 * 60 * 60  # 7-day TTL for TEST rows.


def sort_key(suite, suite_code_hash):
    return f"TEST#{suite}#{suite_code_hash}"


def table_arn():
    account = os.environ["CI_IMAGES_TABLE_ACCOUNT_ID"]
    region = os.environ.get("CI_IMAGES_TABLE_REGION", TABLE_REGION)
    name = os.environ.get("CI_IMAGES_TABLE_NAME", TABLE_NAME)
    return f"arn:aws:dynamodb:{region}:{account}:table/{name}"


def _client():
    region = os.environ.get("CI_IMAGES_TABLE_REGION", TABLE_REGION)
    return boto3.client("dynamodb", region_name=region)


def check_test_skip(image_content_hash, suite, suite_code_hash, client=None):
    """Return True iff this suite may be skipped (a matching PASS row exists)."""
    client = client or _client()
    sk = sort_key(suite, suite_code_hash)
    try:
        resp = client.get_item(
            TableName=table_arn(),
            Key={
                "image_content_hash": {"S": image_content_hash},
                "sort_key": {"S": sk},
            },
            ConsistentRead=True,
        )
    except (ClientError, BotoCoreError) as e:
        LOG.warning("test-skip read failed (%s); running suite %s", e, suite)
        return False

    hit = "Item" in resp
    LOG.info(
        "test-skip %s for suite=%s hash=%s sk=%s",
        "HIT (skip)" if hit else "MISS (run)",
        suite,
        image_content_hash,
        sk,
    )
    return hit


def record_test_pass(image_content_hash, suite, suite_code_hash, client=None, now=None):
    """Write the PASS row for a fully-passed suite."""
    client = client or _client()
    sk = sort_key(suite, suite_code_hash)
    now = now if now is not None else time.time()
    passed_at = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    ttl = int(now) + TEST_ROW_TTL_SECONDS
    client.put_item(
        TableName=table_arn(),
        Item={
            "image_content_hash": {"S": image_content_hash},
            "sort_key": {"S": sk},
            "passed_at": {"S": passed_at},
            "ttl": {"N": str(ttl)},
        },
    )
    LOG.info("recorded PASS for suite=%s hash=%s sk=%s", suite, image_content_hash, sk)


def main(argv=None):
    parser = argparse.ArgumentParser(description="dlc-ci-images test-skip cache read/write.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image-content-hash", required=True)
    common.add_argument("--suite", required=True)
    common.add_argument("--suite-code-hash", required=True)

    sub.add_parser("check", parents=[common], help="Exit 0 to skip, 1 to run.")
    sub.add_parser("record", parents=[common], help="Record a PASS row.")

    args = parser.parse_args(argv)

    if args.command == "check":
        skip = check_test_skip(args.image_content_hash, args.suite, args.suite_code_hash)
        return 0 if skip else 1

    if args.command == "record":
        record_test_pass(args.image_content_hash, args.suite, args.suite_code_hash)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
