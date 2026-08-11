"""Unit tests for the dlc-ci-images test-skip store read/write."""

import importlib.util
import os
from pathlib import Path
from unittest import mock

import boto3
import pytest
from moto import mock_aws

MODULE_PATH = Path(__file__).resolve().parent.parent / "ci_images_store.py"
spec = importlib.util.spec_from_file_location("store", MODULE_PATH)
store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(store)

HASH = "sha256:abc123"
SUITE = "pytorch/single_gpu"
CODE_HASH = "sha256:def456"


@pytest.fixture
def dynamo():
    """A moto DynamoDB with the cache table, wired so table_arn() resolves to it."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name=store.TABLE_REGION)
        with mock.patch.dict(os.environ, {"CI_IMAGES_TABLE_ACCOUNT_ID": "123456789012"}):
            client.create_table(
                TableName=store.TABLE_NAME,
                AttributeDefinitions=[
                    {"AttributeName": "image_content_hash", "AttributeType": "S"},
                    {"AttributeName": "sort_key", "AttributeType": "S"},
                ],
                KeySchema=[
                    {"AttributeName": "image_content_hash", "KeyType": "HASH"},
                    {"AttributeName": "sort_key", "KeyType": "RANGE"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            yield client


def test_sort_key_format():
    assert store.sort_key("sanity", "sha256:7c1e") == "TEST#sanity#sha256:7c1e"


def test_miss_when_no_row(dynamo):
    assert store.check_test_skip(HASH, SUITE, CODE_HASH, client=dynamo) is False


def test_record_then_hit(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    assert store.check_test_skip(HASH, SUITE, CODE_HASH, client=dynamo) is True


def test_hit_requires_matching_code_hash(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    assert store.check_test_skip(HASH, SUITE, "sha256:other", client=dynamo) is False


def test_hit_requires_matching_image_hash(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    assert store.check_test_skip("sha256:different", SUITE, CODE_HASH, client=dynamo) is False


def test_check_cli_exits_zero_on_hit(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    argv = ["check", "--image-content-hash", HASH, "--suite", SUITE, "--suite-code-hash", CODE_HASH]
    assert store.main(argv) == 0


def test_check_cli_exits_nonzero_on_miss(dynamo):
    argv = ["check", "--image-content-hash", HASH, "--suite", SUITE, "--suite-code-hash", CODE_HASH]
    assert store.main(argv) != 0
