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
    "gcc-7:7.5.0-3ubuntu1~18.04": [
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-13844",
            "NotApplicableOn": [
                {
                    "framework": ["tensorflow"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu", "gpu", "eia", "neuron"],
                },
                {
                    "framework": ["pytorch"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu", "gpu", "eia", "neuron"],
                },
                {
                    "framework": ["mxnet"],
                    "image_type": ["training", "inference"],
                    "python_version": ["py36", "py37", "py38"],
                    "device_type": ["cpu", "gpu", "eia", "neuron"],
                },
            ],
        },
    ],
    "gcc-8:8.4.0-1ubuntu1~18.04": {
        {
            "Severity": "MEDIUM",
            "CVE": "CVE-2020-13844",

        },
    },
}


def
