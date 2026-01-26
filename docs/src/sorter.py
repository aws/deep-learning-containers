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
"""Sorting tiebreaker functions for image tables."""

from constants import GLOBAL_CONFIG


def repository_sorter(img) -> int:
    """Repository order: by table_order index."""
    table_order = GLOBAL_CONFIG.get("table_order", [])
    try:
        return table_order.index(img.repository)
    except ValueError:
        return len(table_order)


def platform_sorter(img) -> int:
    """Platform order: SageMaker before EC2."""
    return 0 if img.get("platform") == "sagemaker" else 1


def accelerator_sorter(img) -> int:
    """Accelerator order: GPU before NeuronX before CPU."""
    return {"gpu": 0, "neuronx": 1, "cpu": 2}.get(img.get("accelerator", "").lower(), 3)
