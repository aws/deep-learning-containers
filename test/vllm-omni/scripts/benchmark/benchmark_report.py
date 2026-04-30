#!/usr/bin/env python3
"""
Aggregate vLLM-Omni benchmark artifact directories into a markdown report.

Reads from `results/omni-benchmark-*/benchmark_results/*.json` (as produced by
the dispatch-vllm-omni-benchmark.yml workflow) and prints a markdown table to
stdout suitable for $GITHUB_STEP_SUMMARY.

Usage:
    python3 benchmark_report.py <results_root>
"""

import glob
import json
import os
import sys


def main(root: str) -> int:
    print("# vLLM-Omni Benchmark Report\n")
    print(
        "| Model / Fleet | Status | Req/s | Throughput | p50 E2E (ms) | p95 E2E (ms) | p99 E2E (ms) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")

    for artifact_dir in sorted(glob.glob(os.path.join(root, "omni-benchmark-*"))):
        name = os.path.basename(artifact_dir).replace("omni-benchmark-", "")
        # The upload-artifact action puts files under the artifact name.
        # Workflow writes results to $RESULTS_DIR so JSONs should be at the top
        # of artifact_dir (when upload-artifact uploads benchmark_results/ directly).
        jsons = sorted(glob.glob(os.path.join(artifact_dir, "**", "*.json"), recursive=True))
        if not jsons:
            print(f"| {name} | NO_RESULT | - | - | - | - | - |")
            continue

        with open(jsons[0]) as f:
            doc = json.load(f)
        summary = doc.get("summary", doc)

        failed = summary.get("failed", 0) or 0
        status = "FAIL" if failed else "PASS"

        rps = summary.get("requests_per_second", "-")
        # Pick the most informative throughput metric available
        tput = (
            summary.get("output_tokens_per_second")  # chat
            or summary.get("audio_throughput_s_per_s")  # tts
            or summary.get("images_per_second")  # image
            or summary.get("videos_per_second")  # video
            or "-"
        )
        # Chat-omni adds TTFT; surface it when present
        ttft_p95 = summary.get("ttft_ms", {}).get("p95")
        e2e = summary.get("e2e_ms", {}) or {}
        p50 = e2e.get("median", "-")
        p95 = e2e.get("p95", "-")
        p99 = e2e.get("p99", "-")
        extra = f" (TTFT p95 {ttft_p95} ms)" if ttft_p95 is not None else ""

        print(f"| {name} | {status} | {rps} | {tput} | {p50}{extra} | {p95} | {p99} |")

    return 0  # report is informational; workflow already fails individual jobs on threshold miss


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: benchmark_report.py <results_root>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
