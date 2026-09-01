"""dlc-ci-images test-skip cache: read (check skip) and write (record pass).

image_content_hash = sha256 over the ordered rootfs diff_ids + the OCI image
config object, read from the pushed image's registry config. suite_code_hash
= sha256 over the declared test files in test-suites.yml.

Every OCI image config carries three required top-level fields: ``architecture``, ``os``,
and ``rootfs``. The rest are optional and appear only when non-empty -- the ``config`` object
(runtime execution parameters), plus ``created``, ``history`` (both non-deterministic) and others.
For more info, see: https://github.com/opencontainers/image-spec/blob/main/config.md#properties

Both a filesystem change (RUN/COPY/ADD) and a config change (ENV/LABEL/CMD/...) change the
image_content_hash. We hash only ``diff_ids`` + ``config``; the top-level ``architecture``/``os``
and the non-deterministic ``created`` / ``history`` are excluded along with other optional fields.
The content hash is deterministic today. A build-time ARG (e.g. CACHE_REFRESH, GIT_SHA, *_COMMIT)
is NOT stored in the config -- it reaches the config only if interpolated into an ENV/LABEL after
being declared. Currently in the repo there are no mutable ARGs that are promoted to config values,
but this is not enforced.

Docker Commands
--------------
Changes layer diff_ids: RUN, COPY, ADD, WORKDIR (only if the target path doesn't yet exist)
Changes config: USER, EXPOSE, ENV, ENTRYPOINT, CMD, VOLUME, WORKDIR, LABEL, STOPSIGNAL

Limitations
--------------
- diff_ids are NOT reproducible across an independent (cache-miss) rebuild: BuildKit bakes
  in-layer file mtimes (= build wall-clock) into the layer, and does not normalize them
  unless built with ``rewrite-timestamp=true``. Stable diff_ids come from BuildKit *layer-
  cache reuse* (cache hit -> identical blob -> identical diff_id). A true rebuild re-executes
  the layer and gets fresh mtimes -> a new hash.
- This hashes only what is baked into the image (layers + config). It cannot see
  inputs the image references but does not embed. For example, if a container pulls content
  when it runs (e.g. an entrypoint that runs ``aws s3 cp`` / ``curl``, or a test that downloads
  a model), that content is never in a layer, so a change to it does not change this hash.
  Similarly, build-time fetches can be served from a stale layer cache. ``RUN aws s3 cp ...``
  bakes the bytes into a layer at build time. Since Docker keys its layer cache on the
  *instruction string* and not the fetched content, an unchanged ``RUN`` reuses the old layer
  and never re-pulls. The hash then matches that (stale) image, so the test may skip.
- Test caching is bounded by the daily ``CACHE_REFRESH`` build-arg (see .github/actions/build-image).
  This busts layer caches ~daily, and once the content actually lands in a rebuilt layer the diff_id
  (and the hash) change, so all tests re-run. Additionally, TEST rows have bounded TTL, so prod images
  are only skipped for the length of the row TTL (currently 3 days).
- Files for each test suite do not include shared helpers such as test/test_utils. This is a deliberate
  choice that trades some test coverage for more cache hits. Including shared test infrastructure means
  a change to a test_util invalidates the entire cache. This is adjustable by modifying test-suites.yml.
"""

import logging
import os
import time
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError

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


def check_test_pass(image_content_hash, suite_code_hashes, client=None):
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
