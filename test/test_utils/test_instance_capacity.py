"""Unit tests for the SageMaker instance-pool helpers (CPU-only, no AWS).

These guard the two behaviors that gate every expensive GPU endpoint deploy: the
priority mapping (first candidate becomes priority 1, provisioned first) and the
pool-count bounds SageMaker enforces (1 to 5 pools per variant).
"""

import pytest
from test_utils.instance_capacity import (
    MAX_INSTANCE_POOLS,
    build_instance_pools,
    normalize_instance_types,
)


def test_normalize_str_becomes_single_element_list():
    assert normalize_instance_types("ml.g6.xlarge") == ["ml.g6.xlarge"]


def test_normalize_list_is_passed_through():
    assert normalize_instance_types(["ml.g6.xlarge", "ml.g6.2xlarge"]) == [
        "ml.g6.xlarge",
        "ml.g6.2xlarge",
    ]


def test_ladder_maps_to_ascending_priority_from_one():
    pools = build_instance_pools(["ml.g6.xlarge", "ml.g6.2xlarge", "ml.g6.4xlarge"])
    assert [p.instance_type for p in pools] == [
        "ml.g6.xlarge",
        "ml.g6.2xlarge",
        "ml.g6.4xlarge",
    ]
    assert [p.priority for p in pools] == [1, 2, 3]


def test_single_string_is_one_pool_at_priority_one():
    pools = build_instance_pools("ml.g6.xlarge")
    assert len(pools) == 1
    assert pools[0].instance_type == "ml.g6.xlarge"
    assert pools[0].priority == 1


def test_five_pools_are_accepted():
    pools = build_instance_pools([f"ml.g6.{i}xlarge" for i in range(1, 6)])
    assert len(pools) == MAX_INSTANCE_POOLS
    assert [p.priority for p in pools] == [1, 2, 3, 4, 5]


def test_more_than_five_pools_raises():
    with pytest.raises(ValueError, match="at most 5 instance pools"):
        build_instance_pools([f"ml.g6.{i}xlarge" for i in range(6)])


def test_empty_list_raises():
    with pytest.raises(ValueError, match="at least one instance type"):
        build_instance_pools([])
