"""OpenFold3 SageMaker async-inference integration tests.

One 4-GPU endpoint (ml.g6.12xlarge) serves all cases. The handler's GPU pool
leases one GPU per concurrent request, so "1 GPU" == 1 in-flight request and
"4 GPU" == 4 concurrent requests.

Smoke validation only: a successful prediction returns at least one non-empty
CIF structure. OpenFold3 diffusion is stochastic, so exact-output matching is
not meaningful.
"""

import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

SMALL_QUERY = "small.json"  # ubiquitin, 76 residues
LARGE_QUERY = "large.json"  # synthetic, 622 residues

# concurrency-4 should run in roughly the same wall-time as concurrency-1
# because the 4 requests fan out across the 4 GPUs. Allow generous headroom
# for scheduling/IO jitter; serialized execution would be ~4x.
CONCURRENCY_SPEEDUP_TOLERANCE = 2.0


def _assert_success(result: dict):
    """A successful OpenFold3 prediction has status 'success' and a non-empty CIF."""
    assert result.get("status") == "success", f"prediction failed: {result.get('error', result)}"
    structures = result.get("structures", [])
    assert structures, "no structures returned"
    assert structures[0].get("content"), "first structure has empty CIF content"


def test_small_single(invoke_async):
    """Smoke: small protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = invoke_async([SMALL_QUERY])
    _assert_success(result)
    LOGGER.info(f"small x1 completed in {elapsed:.0f}s")


def test_large_single(invoke_async):
    """Large protein, single request on one GPU returns a valid structure."""
    ((elapsed, result),) = invoke_async([LARGE_QUERY])
    _assert_success(result)
    LOGGER.info(f"large x1 completed in {elapsed:.0f}s")


def test_large_concurrent_uses_gpu_pool(invoke_async):
    """Large protein, 1 request then 4 concurrent: the GPU pool should keep the
    4-concurrent wall-time under ~2x the single-request time (not ~4x)."""
    ((t1, result1),) = invoke_async([LARGE_QUERY])
    _assert_success(result1)

    results4 = invoke_async([LARGE_QUERY] * 4)
    for _, result in results4:
        _assert_success(result)
    t4 = max(elapsed for elapsed, _ in results4)

    LOGGER.info(f"large: 1-request={t1:.0f}s, 4-concurrent(max)={t4:.0f}s, ratio={t4 / t1:.2f}")
    assert t4 < CONCURRENCY_SPEEDUP_TOLERANCE * t1, (
        f"4 concurrent requests took {t4:.0f}s vs {t1:.0f}s for 1 "
        f"(ratio {t4 / t1:.2f} >= {CONCURRENCY_SPEEDUP_TOLERANCE}); GPU pool may not be parallelizing"
    )
