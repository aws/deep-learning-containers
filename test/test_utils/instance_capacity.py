"""Shared SageMaker instance-pool helper for endpoint tests.

SageMaker reports a dry capacity pool per instance type, so a shortage in one size
says nothing about the next size up. An endpoint test that names a single instance
type therefore fails for a reason that has nothing to do with the image under test.

SageMaker's native "instance pools" feature (a.k.a. heterogeneous endpoints) solves
this server-side: a production variant carries a priority-ordered list of up to five
instance types, and SageMaker provisions the highest-priority type available, falling
back to the next on an insufficient-capacity error, all within a single deploy. These
helpers turn a single type or a priority-ordered ladder into that pool list.

Every rung of a ladder must fit the model unaided. Within a family the larger sizes
carry the same single GPU as their xlarge base (L4 24GB for g6, L40S 48GB for g6e), so
they serve identically and differ only in capacity pool. Never ladder across GPU
families — a model sized for L40S will not fit L4.
"""

import logging

from sagemaker.core.shapes import InstancePool

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# SageMaker allows at most five instance types per production variant.
MAX_INSTANCE_POOLS = 5


def normalize_instance_types(instance_type):
    """Accept a single instance type or a priority-ordered list; always return a list.

    Lets a config entry or a parametrize argument stay a plain string when it has no
    fallback, while the deploy path only ever handles one shape.
    """
    if isinstance(instance_type, str):
        return [instance_type]
    return list(instance_type)


def build_instance_pools(instance_type):
    """Turn a single type or a priority-ordered ladder into a list of ``InstancePool``.

    The first candidate gets priority 1 (highest), the next priority 2, and so on.
    SageMaker provisions the highest-priority type with available capacity and falls
    back to lower-priority types on an insufficient-capacity error, so a momentary
    shortage in one pool is absorbed server-side rather than reported as a defect.

    Raises:
        ValueError: if no instance types are supplied, since a variant needs at least
            one pool to provision.
        ValueError: if more than ``MAX_INSTANCE_POOLS`` candidates are supplied, since
            SageMaker rejects a variant with more than five pools.
    """
    types = normalize_instance_types(instance_type)
    if not types:
        raise ValueError("build_instance_pools requires at least one instance type")
    if len(types) > MAX_INSTANCE_POOLS:
        raise ValueError(
            f"SageMaker allows at most {MAX_INSTANCE_POOLS} instance pools per variant; "
            f"got {len(types)}: {types}"
        )
    return [
        InstancePool(instance_type=candidate, priority=priority)
        for priority, candidate in enumerate(types, start=1)
    ]
