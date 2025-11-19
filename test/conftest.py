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
"""Common pytest fixtures for all tests under module test/"""

import logging

import pytest
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.traceback import install


def pytest_addoption(parser):
    parser.addoption("--image-uri", action="store", help="Image URI to be tested")


@pytest.fixture(scope="session")
def image_uri(request):
    return request.config.getoption("--image-uri")


def pytest_configure():
    """
    Set up rich logging for all tests
    To use rich logging in each tests,
    include `logger` fixture in the test argument.
    """
    # Install rich traceback handling
    install(show_locals=True)

    # Custom theme for log levels
    custom_theme = Theme(
        {
            "logging.debug": "grey70",
            "logging.info": "cyan",
            "logging.warning": "yellow",
            "logging.error": "red bold",
            "logging.critical": "red bold reverse",
            "pytest.passed": "green bold",
            "pytest.failed": "red bold",
            "pytest.skipped": "yellow bold",
        }
    )

    # Create console with theme
    console = Console(theme=custom_theme, force_terminal=True, tab_size=2)

    # Configure Rich handler
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_code_width=None,
        tracebacks_show_locals=True,
        show_time=True,
        show_path=True,
        enable_link_path=True,
        markup=True,
    )

    # Set formatter
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    rich_handler.setLevel(logging.DEBUG)

    # Get the root logger and add rich handler
    root_logger = logging.getLogger()
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def logger():
    return logging.getLogger()
