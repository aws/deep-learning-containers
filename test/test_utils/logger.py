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
"""Logging handler for nice formatted logs"""

import logging


# Custom formatter
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": {"text": "\x1b[36;20m", "bold": "\x1b[1;36;20m", "underline": "\x1b[4;36;20m"},
        "INFO": {"text": "\x1b[38;20m", "bold": "\x1b[1;38;20m", "underline": "\x1b[4;38;20m"},
        "WARNING": {"text": "\x1b[33;20m", "bold": "\x1b[1;33;20m", "underline": "\x1b[4;33;20m"},
        "ERROR": {"text": "\x1b[31;20m", "bold": "\x1b[1;31;20m", "underline": "\x1b[4;31;20m"},
        "CRITICAL": {"text": "\x1b[31;1m", "bold": "\x1b[1;31;1m", "underline": "\x1b[4;31;1m"},
        "RESET": "\x1b[0m",
    }

    def format(self, record):
        colors = self.COLORS.get(record.levelname, self.COLORS["DEBUG"])
        reset = self.COLORS["RESET"]

        # Create formatted string with different styles
        format_str = (
            f"{colors['bold']}%(asctime)s{reset} - "
            f"{colors['text']}%(name)s{reset} - "
            f"{colors['underline']}%(levelname)s{reset} - "
            f"{colors['text']}%(message)s{reset}"
        )
        formatter = logging.Formatter(format_str)
        return formatter.format(record)
