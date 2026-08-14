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
OTHER_SUITE = "sanity"
OTHER_CODE_HASH = "sha256:ghi789"


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


def _get_row(client):
    return client.get_item(
        TableName=store.table_arn(),
        Key={
            "image_content_hash": {"S": HASH},
            "sort_key": {"S": store.sort_key(SUITE, CODE_HASH)},
        },
    )["Item"]


def test_sort_key_format():
    assert store.sort_key("sanity", "sha256:7c1e") == "TEST#sanity#sha256:7c1e"


def test_record_writes_ci_image_tag_attribute(dynamo):
    ci_image_tag = "sglang-ec2-amzn2023-0.5.12.dlc1-gpu-py312-cu130-pr-123"
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo, ci_image_tag=ci_image_tag)
    assert _get_row(dynamo)["ci_image_tag"]["S"] == ci_image_tag


def test_record_omits_ci_image_tag_when_not_given(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    assert "ci_image_tag" not in _get_row(dynamo)


def test_record_omits_ci_image_tag_when_empty_string(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo, ci_image_tag="")
    assert "ci_image_tag" not in _get_row(dynamo)


def test_check_test_skip_empty_input_returns_empty(dynamo):
    assert store.check_test_skip(HASH, {}, client=dynamo) == set()


def test_check_test_skip_all_miss_returns_empty(dynamo):
    result = store.check_test_skip(HASH, {SUITE: CODE_HASH, OTHER_SUITE: OTHER_CODE_HASH}, client=dynamo)
    assert result == set()


def test_check_test_skip_returns_only_recorded_suites(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    # OTHER_SUITE is not recorded, so only SUITE should come back as a hit.
    result = store.check_test_skip(HASH, {SUITE: CODE_HASH, OTHER_SUITE: OTHER_CODE_HASH}, client=dynamo)
    assert result == {SUITE}


def test_check_test_skip_returns_multiple_hits(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    store.record_test_pass(HASH, OTHER_SUITE, OTHER_CODE_HASH, client=dynamo)
    result = store.check_test_skip(HASH, {SUITE: CODE_HASH, OTHER_SUITE: OTHER_CODE_HASH}, client=dynamo)
    assert result == {SUITE, OTHER_SUITE}


def test_check_test_skip_respects_code_hash(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    # Same suite, different code hash -> no hit.
    result = store.check_test_skip(HASH, {SUITE: "sha256:changed"}, client=dynamo)
    assert result == set()


def test_check_test_skip_respects_image_hash(dynamo):
    store.record_test_pass(HASH, SUITE, CODE_HASH, client=dynamo)
    result = store.check_test_skip("sha256:different", {SUITE: CODE_HASH}, client=dynamo)
    assert result == set()
