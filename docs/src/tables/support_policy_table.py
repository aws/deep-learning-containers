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
"""Support policy table generation."""

from utils import render_table

COLUMNS = ["Framework", "Version", "CUDA", "GA Date", "EOP Date"]


def generate(yaml_data: dict) -> str:
    """Generate supported and unsupported framework tables."""
    policy = yaml_data.get("support_policy", {})
    sections = []

    # Supported frameworks table
    supported = policy.get("supported", {})
    if supported:
        rows = []
        for framework, versions in supported.items():
            for version, info in versions.items():
                rows.append(
                    [
                        framework.title(),
                        version,
                        info.get("cuda", ""),
                        info.get("ga", ""),
                        info.get("eop", ""),
                    ]
                )
        sections.append("## Supported Frameworks\n\n" + render_table(COLUMNS, rows))

    # Unsupported frameworks table
    unsupported = policy.get("unsupported", {})
    if unsupported:
        rows = []
        for framework, versions in unsupported.items():
            for version, info in versions.items():
                rows.append(
                    [
                        framework.title(),
                        version,
                        info.get("cuda", ""),
                        info.get("ga", ""),
                        info.get("eop", ""),
                    ]
                )
        sections.append("## Unsupported Frameworks\n\n" + render_table(COLUMNS, rows))

    return "\n\n".join(sections)
