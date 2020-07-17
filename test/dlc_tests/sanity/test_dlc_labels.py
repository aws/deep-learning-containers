import boto3
import pytest


def test_dlc_major_version_label(image, region):
    ecr_client = boto3.client("ecr", region_name=region)
