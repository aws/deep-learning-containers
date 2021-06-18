import json

from datetime import datetime
from time import sleep, time

import pytest

from packaging.version import Version

from invoke import run
from invoke import Context

from test.test_utils import get_account_id_from_image_uri, get_region_from_image_uri, login_to_ecr_registry
from test.test_utils import ecr as ecr_utils
from test.test_utils.security import CVESeverity


FALSE_ALARM_CVEs = {
    "avahi:0.7-3.1ubuntu1.2": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2021-3468",
            "NotApplicableOn": [
                {
                    "framework": "tensorflow",
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Deferred",
                },
            ],
        },
    ],
    "gcc-7:7.5.0-3ubuntu1~18.04": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-13844",
            "NotApplicableOn": [
                {
                    "framework": "tensorflow",
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Only applicable to ARM arch",
                },
            ],
        },
    ],
    "gcc-8:8.4.0-1ubuntu1~18.04": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-13844",
            "NotApplicableOn": [
                {
                    "framework": "tensorflow",
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Only applicable to ARM arch",
                },
            ],
        },
    ],
    "gcc-defaults:1.176ubuntu2.3": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-13844",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Only applicable to ARM arch",
                },
            ],
        },
    ],
    "imagemagick:8:6.9.7.4+dfsg-16ubuntu6.11": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-27752",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Deferred",
                },
            ],
        },
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-25664",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Deferred",
                },
            ],
        },
    ],
    "krb5:1.16-2ubuntu0.2": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2018-20217",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Needed",
                },
            ],
        },
    ],
    "nghttp2:1.30.0-1ubuntu1": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2019-9511",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Needed",
                },
            ],
        },
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2019-9513",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Needed",
                },
            ],
        },
    ],
    "sqlite3:3.22.0-1ubuntu0.4": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-9794",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Deferred",
                },
            ],
        },
    ],
    "systemd:237-3ubuntu10.48": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-13529",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Deferred",
                },
            ],
        },
    ],
    "wget:1.19.4-1ubuntu2.2": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2021-31879",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "framework_version": ["2.4.1"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu"],
                    "reason": "Deferred",
                },
            ],
        },
    ],
}


def compare_to_false_alarm_list():
    pass
