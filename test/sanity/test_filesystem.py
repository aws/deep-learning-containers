# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Sanity tests for DLC container filesystem hygiene"""

import calendar
import logging
import os
import re
import shutil
import subprocess
import time
from pprint import pformat

import pytest

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

STRAY_ARTIFACTS = [r"\.py$"]


def _assert_no_stray_artifacts(path):
    """Assert no stray artifact patterns found in directory listing."""
    if not os.path.exists(path):
        return
    entries = os.listdir(path)
    for entry in entries:
        for pattern in STRAY_ARTIFACTS:
            assert not re.search(pattern, entry), (
                f"Stray artifact '{entry}' matching '{pattern}' found in {path}"
            )


def test_stray_files():
    """Ensure no stray build artifacts exist in key directories."""
    for path in ["/tmp", "/var/tmp", os.path.expanduser("~"), "/"]:
        _assert_no_stray_artifacts(path)


def test_tmp_dir_is_clean():
    """/tmp should not contain unexpected files."""
    for f in os.listdir("/tmp/"):
        if (
            f.startswith(".")
            or "system" in f.lower()
            or "dkms" in f.lower()
            or "hsperfdata" in f
        ):
            continue
        pytest.fail(f"/tmp contains unexpected file: {f}")


def test_var_tmp_is_empty():
    """/var/tmp should be empty."""
    contents = os.listdir("/var/tmp")
    assert not contents, f"/var/tmp is not empty: {pformat(contents)}"


def test_cache_dir_is_clean():
    """Cache directory should not contain unexpected files."""
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    if not os.path.exists(cache_dir):
        return
    contents = os.listdir(cache_dir)
    LOGGER.info("Contents of cache directory: %s", contents)
    for item in contents:
        assert item.startswith("pip"), f"Unexpected file in cache dir: {item}"


def test_no_viminfo():
    """Vim info file should not exist."""
    viminfo = os.path.join(os.path.expanduser("~"), ".viminfo")
    assert not os.path.exists(viminfo), f"{viminfo} still exists"


def test_bash_history_is_empty():
    """Bash history should be empty."""
    history = os.path.join(os.path.expanduser("~"), ".bash_history")
    if os.path.exists(history):
        with open(history) as f:
            assert not f.read(), f"{history} contains history"


def test_no_files_modified_before_boot():
    """No history files in home or cloud-init files should predate boot time."""
    with open("/proc/uptime") as f:
        uptime_seconds = int(round(float(f.readline().split()[0])))
    boot_time = int(calendar.timegm(time.gmtime())) - uptime_seconds

    checks = [
        (os.path.expanduser("~"), "history", False),
        ("/var/lib/cloud/instances/", None, True),
    ]
    for folder, mask, recursive in checks:
        if not os.path.exists(folder):
            continue
        if recursive:
            files = [
                os.path.join(dp, f)
                for dp, _, filenames in os.walk(folder)
                for f in filenames
            ]
        else:
            files = [
                f
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            ]
        if mask:
            files = [f for f in files if mask in f]
        for filepath in files:
            mtime = int(round(os.path.getmtime(filepath)))
            assert mtime >= boot_time, f"{filepath} was modified before boot"


def test_repo_anaconda_not_present():
    """All installed conda packages should not come from repo.anaconda.com."""
    if not shutil.which("conda"):
        pytest.skip("conda is not installed, skipping test")

    result = subprocess.run(
        ["conda", "list", "--explicit"],
        capture_output=True,
        text=True,
        check=True,
    )
    offending = [
        line for line in result.stdout.splitlines() if "repo.anaconda.com" in line
    ]
    assert not offending, (
        f"Packages installed from repo.anaconda.com found:\n{pformat(offending)}"
    )