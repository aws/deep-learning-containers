"""Shared SageMaker instance-capacity fallback for endpoint tests.

SageMaker reports a dry capacity pool per instance type, so a shortage in one size
says nothing about the next size up. An endpoint test that names a single instance
type therefore fails for a reason that has nothing to do with the image under test.

These helpers support both the legacy client-side fallback loop and SageMaker's native
instance pools (a.k.a. heterogeneous endpoints). Native pools put the priority-ordered
instance-type ladder on one production variant, so SageMaker performs capacity fallback
server-side within a single deploy.

Every rung of a ladder must fit the model unaided. Within a family the larger sizes
carry the same single GPU as their xlarge base (L4 24GB for g6, L40S 48GB for g6e), so
they serve identically and differ only in capacity pool. Never ladder across GPU
families — a model sized for L40S will not fit L4.
"""

import logging
import os

from sagemaker.core.shapes import InstancePool, ProductionVariantCapacityReservationConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# SageMaker allows at most five instance types per production variant.
MAX_INSTANCE_POOLS = 5

# An ML Reservation covers exactly one instance type. Both are set by CI; unset means
# every endpoint provisions from the open pool, which is the pre-reservation behavior.
ML_RESERVATION_ARN = os.environ.get("ML_RESERVATION_ARN", "").strip()
ML_RESERVATION_INSTANCE_TYPE = os.environ.get(
    "ML_RESERVATION_INSTANCE_TYPE", "ml.g6.xlarge"
).strip()

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


def build_instance_pools(instance_type):
    """Turn a single type or priority-ordered ladder into native SageMaker pools.

    The first candidate gets priority 1 (highest), the next priority 2, and so on.
    SageMaker provisions the highest-priority type with available capacity and falls
    back to lower-priority types on an insufficient-capacity error.

    Raises:
        ValueError: if no instance types are supplied.
        ValueError: if more than ``MAX_INSTANCE_POOLS`` candidates are supplied.
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


def ml_reservation_config(instance_type):
    """Reservation config for this ladder, or None to provision from the open pool.

    Returns None unless an ARN is set and the reserved type leads the ladder. A
    reservation for a type we are not asking for first cannot be honored, so attaching
    it only adds a way for the deploy to go wrong.

    ``capacity_reservation_preference`` is deliberately left unset. Its only valid value
    is ``capacity-reservations-only``, which turns a busy reservation into a hard deploy
    failure; unset keeps the instance-pool ladder as fallback.
    """
    if not ML_RESERVATION_ARN:
        return None

    types = normalize_instance_types(instance_type)
    if not types or types[0] != ML_RESERVATION_INSTANCE_TYPE:
        LOGGER.info(
            f"[capacity] reservation covers {ML_RESERVATION_INSTANCE_TYPE}, ladder leads "
            f"with {types[0] if types else 'nothing'}; using open pool"
        )
        return None

    LOGGER.info(f"[capacity] using ML Reservation: {ML_RESERVATION_ARN}")
    return ProductionVariantCapacityReservationConfig(ml_reservation_arn=ML_RESERVATION_ARN)


def log_capacity_placement(endpoint, label=""):
    """Log the instance type an InService endpoint landed on and whether it was reserved.

    Sending the reservation config only proves what we asked for. SageMaker reports what
    it did in ``used_by_current_endpoint``, without which a healthy open pool looks
    identical to a working reservation. Best-effort: this must never fail a passing test.
    """
    try:
        variant = (endpoint.refresh() or endpoint).production_variants[0]
        pools = [
            f"{p.instance_type} x{p.current_instance_count}"
            for p in variant.instance_pools or []
            if p.current_instance_count
        ]
        reservation = variant.capacity_reservation_config
        held = f"reserved={reservation.used_by_current_endpoint}" if reservation else "open pool"
        LOGGER.info(f"[capacity] {label} on {pools or 'unreported'}, {held}")
    except Exception as e:
        LOGGER.warning(f"[capacity] {label} placement unknown: {e}")


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
