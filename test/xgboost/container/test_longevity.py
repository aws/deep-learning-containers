"""Longevity / resource-impact container test.

Exercises the XGBoost serving container under sustained inference load and
tracks its resource footprint (memory + CPU) over time. The goal is to catch
container-level regressions that unit / functional tests miss:

- memory leaks — RSS that climbs steadily and never plateaus under repeated
  ``/invocations`` requests
- runaway resource use — CPU or memory that blows past a sane bound
- loss of liveness — the container becoming unresponsive after prolonged load

This is intentionally lightweight (a couple of minutes) so it can run on every
PR alongside the other local container tests, not just on the release pipeline.
"""

import logging
import os
import time

from .container_helper import ServingContainer

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Total wall-clock duration to keep hammering the container.
LOAD_DURATION_SECONDS = int(os.environ.get("XGB_LONGEVITY_DURATION", "150"))
# How often to sample docker stats (memory / cpu).
SAMPLE_INTERVAL_SECONDS = 5
# Warm-up window ignored when computing the memory-growth baseline — lets the
# process finish lazy allocation / model load before we start trusting numbers.
WARMUP_SECONDS = 20

# Memory growth from the post-warmup baseline to the end of the run that we
# consider indicative of a leak. XGBoost inference on a fixed model should be
# flat; a steadily climbing RSS is the signal we want to catch.
MEM_GROWTH_LIMIT_MB = 150
# Absolute ceiling — the small mnist model should never need anywhere near this.
MEM_ABSOLUTE_LIMIT_MB = 4096

MODEL_NAME = "mnist-xgb-model"
INPUT_FILE = "mnist-1.csv"
CONTENT_TYPE = "text/csv"


def _model_path(resources, model_name):
    return os.path.join(resources, "models", model_name)


def _input_path(resources, filename):
    return os.path.join(resources, "input", filename)


def test_serving_longevity_no_memory_leak(docker_client, image_uri, inference_resources):
    """Sustained-load longevity check on the serving container.

    Sends inference requests continuously for ``LOAD_DURATION_SECONDS`` while
    sampling memory and CPU. Asserts the container stays healthy, every request
    succeeds, and memory does not grow past the leak threshold.
    """
    model_dir = _model_path(inference_resources, MODEL_NAME)
    with open(_input_path(inference_resources, INPUT_FILE), "rb") as f:
        payload = f.read()

    request_count = 0
    failures = 0
    samples = []  # list of (elapsed_s, mem_mb, cpu_percent)

    with ServingContainer(docker_client, image_uri, model_dir) as ctx:
        start = time.time()
        next_sample = 0.0

        while True:
            elapsed = time.time() - start
            if elapsed >= LOAD_DURATION_SECONDS:
                break

            # Drive load.
            try:
                resp = ctx.invocations(data=payload, content_type=CONTENT_TYPE)
                request_count += 1
                if resp.status_code != 200:
                    failures += 1
                    LOGGER.warning("Non-200 response: %s %s", resp.status_code, resp.text[:200])
            except Exception as exc:  # connection dropped == container in trouble
                failures += 1
                LOGGER.warning("Invocation raised: %s", exc)

            # Sample resources on a fixed cadence.
            if elapsed >= next_sample:
                sample = ctx.stats()
                samples.append((elapsed, sample["mem_mb"], sample["cpu_percent"]))
                LOGGER.info(
                    "t=%5.1fs reqs=%d mem=%.1fMB cpu=%.1f%%",
                    elapsed,
                    request_count,
                    sample["mem_mb"],
                    sample["cpu_percent"],
                )
                next_sample = elapsed + SAMPLE_INTERVAL_SECONDS

        # Container must still answer a health check after the load.
        assert ctx.ping().status_code == 200, "container unhealthy after sustained load"

    # -- assertions ----------------------------------------------------------

    assert request_count > 0, "no requests were sent"
    assert samples, "no resource samples collected"
    assert failures == 0, f"{failures}/{request_count} invocations failed under sustained load"

    mem_series = [mem for _, mem, _ in samples]
    peak_mem = max(mem_series)
    cpu_series = [cpu for _, _, cpu in samples]
    peak_cpu = max(cpu_series)

    # Baseline = first sample taken after the warm-up window (fall back to the
    # first sample if the run is shorter than warm-up for any reason).
    post_warmup = [mem for elapsed, mem, _ in samples if elapsed >= WARMUP_SECONDS]
    baseline_mem = post_warmup[0] if post_warmup else mem_series[0]
    final_mem = mem_series[-1]
    mem_growth = final_mem - baseline_mem

    LOGGER.info(
        "longevity summary: requests=%d duration=%ds peak_mem=%.1fMB "
        "baseline_mem=%.1fMB final_mem=%.1fMB growth=%.1fMB peak_cpu=%.1f%%",
        request_count,
        LOAD_DURATION_SECONDS,
        peak_mem,
        baseline_mem,
        final_mem,
        mem_growth,
        peak_cpu,
    )

    assert peak_mem < MEM_ABSOLUTE_LIMIT_MB, (
        f"peak memory {peak_mem:.1f}MB exceeded absolute ceiling {MEM_ABSOLUTE_LIMIT_MB}MB"
    )
    assert mem_growth < MEM_GROWTH_LIMIT_MB, (
        f"memory grew {mem_growth:.1f}MB (baseline {baseline_mem:.1f}MB -> "
        f"final {final_mem:.1f}MB) over {LOAD_DURATION_SECONDS}s, exceeding leak "
        f"threshold {MEM_GROWTH_LIMIT_MB}MB"
    )
