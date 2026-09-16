"""Shared SageMaker instance-capacity fallback for endpoint tests.

SageMaker reports a dry capacity pool per instance type, so a shortage in one size
says nothing about the next size up. An endpoint test that names a single instance
type therefore fails for a reason that has nothing to do with the image under test.

These helpers let a test declare a priority-ordered ladder of instance types and walk
it, so a momentary shortage in one pool is absorbed rather than reported as a defect.
An exhausted ladder raises: skipping would leave the model unvalidated while the run
still reported green, hiding a real coverage gap.

Every rung of a ladder must fit the model unaided. Within a family the larger sizes
carry the same single GPU as their xlarge base (L4 24GB for g6, L40S 48GB for g6e), so
they serve identically and differ only in capacity pool. Never ladder across GPU
families — a model sized for L40S will not fit L4.
"""

import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# SageMaker reports a dry capacity pool as one of these. None of them indicate an image
# or test defect, so they trigger instance-type fallback instead of failing the suite.
CAPACITY_TOKENS = ("InsufficientInstanceCapacity", "ResourceLimitExceeded", "CapacityError")


def is_capacity_error(exc):
    """True if a deploy failed for lack of instance capacity, not a real defect."""
    return any(token.lower() in str(exc).lower() for token in CAPACITY_TOKENS)


def normalize_instance_types(instance_type):
    """Accept a single instance type or a priority-ordered list; always return a list.

    Lets a config entry or a parametrize argument stay a plain string when it has no
    fallback, while the deploy path only ever handles one shape.
    """
    if isinstance(instance_type, str):
        return [instance_type]
    return list(instance_type)


def deploy_with_capacity_fallback(instance_type, deploy, label):
    """Call ``deploy(instance_type)`` down the ladder; return the first success.

    ``deploy`` takes one instance type and returns whatever the caller needs. It MUST
    clean up its own partially-created resources before raising, otherwise each dry
    candidate leaks a Model, an EndpointConfig, and a Failed Endpoint.

    A non-capacity exception propagates immediately, so a genuine image or config
    defect is never masked as a capacity shortage. ``label`` names the thing being
    deployed, for logs and the final error.

    No sleep between candidates: SageMaker already retries the pool internally for
    several minutes before returning a capacity error, so each attempt carries its own
    backoff.

    Raises:
        AssertionError: if every candidate in the ladder is out of capacity.
    """
    candidates = normalize_instance_types(instance_type)

    last_error = None
    for candidate in candidates:
        try:
            return deploy(candidate)
        except Exception as e:
            if not is_capacity_error(e):
                raise  # a real deploy failure — surface it
            last_error = e
            LOGGER.warning(f"[capacity] no {candidate} capacity for {label}: {e}")
            continue

    raise AssertionError(
        f"No instance capacity for {label} on any of {candidates} (ICE). Last error: {last_error}"
    )
