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
# 3 day TTL for TEST rows for unforeseen edge cases. Layer caches are currently
# set to refresh every 1 day in .github/actions/build-image/action.yml.
# Test caches will thus be available for min{CACHE_REFRESH, TEST_ROW_TTL} = 1 day.
TEST_ROW_TTL_SECONDS = 3 * 24 * 60 * 60


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


def check_test_skip(image_content_hash, suite_code_hashes, client=None):
    """Return the set of suites that may be skipped, in one BatchGetItem.

    ``suite_code_hashes`` maps suite name -> its suite_code_hash. A suite is in
    the returned set iff a matching PASS row exists.
    """
    if not suite_code_hashes:
        return set()
    client = client or _client()
    arn = table_arn()
    sk_to_suite = {sort_key(s, h): s for s, h in suite_code_hashes.items()}
    keys = [
        {"image_content_hash": {"S": image_content_hash}, "sort_key": {"S": sk}}
        for sk in sk_to_suite
    ]
    try:
        resp = client.batch_get_item(RequestItems={arn: {"Keys": keys, "ConsistentRead": True}})
    except (ClientError, BotoCoreError) as e:
        LOG.warning("batch test-skip read failed (%s); running all suites", e)
        return set()

    hits = {item["sort_key"]["S"] for item in resp.get("Responses", {}).get(arn, [])}
    skippable = {sk_to_suite[sk] for sk in hits if sk in sk_to_suite}
    LOG.info(
        "batch test-skip: %d/%d suites skippable for hash=%s",
        len(skippable),
        len(suite_code_hashes),
        image_content_hash,
    )
    return skippable


def record_test_pass(
    image_content_hash, suite, suite_code_hash, client=None, now=None, ci_image_tag=None
):
    """Write the PASS row for a fully-passed suite."""
    client = client or _client()
    sk = sort_key(suite, suite_code_hash)
    now = now if now is not None else time.time()
    passed_at = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    ttl = int(now) + TEST_ROW_TTL_SECONDS
    item = {
        "image_content_hash": {"S": image_content_hash},
        "sort_key": {"S": sk},
        "passed_at": {"S": passed_at},
        "ttl": {"N": str(ttl)},
    }
    if ci_image_tag:
        item["ci_image_tag"] = {"S": ci_image_tag}
    client.put_item(TableName=table_arn(), Item=item)
    LOG.info("recorded PASS for suite=%s hash=%s sk=%s", suite, image_content_hash, sk)


def main(argv=None):
    parser = argparse.ArgumentParser(description="dlc-ci-images test-skip cache read/write.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--image-content-hash", required=True)
    common.add_argument("--suite", required=True)
    common.add_argument("--suite-code-hash", required=True)

    record = sub.add_parser("record", parents=[common], help="Record a PASS row.")
    record.add_argument(
        "--ci-image-tag",
        default=None,
        help="CI image tag for the image build being tested.",
    )

    args = parser.parse_args(argv)

    if args.command == "record":
        record_test_pass(
            args.image_content_hash,
            args.suite,
            args.suite_code_hash,
            ci_image_tag=args.ci_image_tag,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
