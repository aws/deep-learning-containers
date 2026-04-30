#!/usr/bin/env python3
"""
Streaming chat-omni benchmark client for vLLM-Omni /v1/chat/completions.

Measures TTFT, TPOT, ITL, E2E, and tokens/sec — matching the metrics reported
by `vllm bench serve --omni --backend openai-chat-omni` so numbers are
comparable to upstream benchmarks.

Protocol: OpenAI chat completions with `stream=true` (SSE).

Usage:
    python chat_omni_benchmark_client.py \\
        --base-url http://localhost:8080 \\
        --num-prompts 16 --concurrency 2 \\
        --max-tokens 128 \\
        --output-json results.json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

DEFAULT_PROMPTS = [
    "Explain photosynthesis in two short paragraphs.",
    "What is the difference between asynchronous and synchronous programming?",
    "Summarize the plot of a short novel in 150 words.",
    "Compose a four-line limerick about a data center.",
    "Describe how a microwave oven heats food at the molecular level.",
    "List five tips to improve the readability of Python code.",
    "What are three common causes of tail latency in distributed systems?",
    "Write a short user-facing error message for a failed file upload.",
    "Describe the steps of the TCP three-way handshake.",
    "Give a beginner-friendly introduction to vector databases.",
]


def parse_sse_chunk(line: bytes):
    """Parse one SSE data line. Returns (done, delta_text, usage_or_none)."""
    if not line.startswith(b"data:"):
        return False, "", None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return True, "", None
    try:
        doc = json.loads(payload)
    except json.JSONDecodeError:
        return False, "", None
    usage = doc.get("usage")
    choices = doc.get("choices") or []
    if not choices:
        return False, "", usage
    delta = choices[0].get("delta") or {}
    return False, delta.get("content") or "", usage


async def send_request(session, url, payload, req_id):
    """Stream a chat completion. Returns timing + token stats."""
    t0 = time.perf_counter()
    ttft = None
    token_times = []  # perf_counter timestamps of each token chunk
    output_text_parts = []
    usage = None

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "req_id": req_id,
                    "ok": False,
                    "status": resp.status,
                    "error": body[:300],
                    "e2e_ms": (time.perf_counter() - t0) * 1000,
                }
            async for line in resp.content:
                line = line.strip()
                if not line:
                    continue
                done, delta, u = parse_sse_chunk(line)
                if u is not None:
                    usage = u
                if delta:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    token_times.append(now)
                    output_text_parts.append(delta)
                if done:
                    break
    except Exception as e:
        return {
            "req_id": req_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "e2e_ms": (time.perf_counter() - t0) * 1000,
        }

    e2e = time.perf_counter() - t0
    output_text = "".join(output_text_parts)

    # Token counts: prefer usage from the server; otherwise count chunks (rough).
    output_tokens = None
    if usage and isinstance(usage, dict):
        output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = len(token_times)

    # TPOT (ms) = (e2e - ttft) / (output_tokens - 1)
    tpot_ms = None
    if ttft is not None and output_tokens and output_tokens > 1:
        tpot_ms = ((e2e - ttft) / (output_tokens - 1)) * 1000

    # Per-token inter-arrival times (ms), after the first
    itls = [(token_times[i] - token_times[i - 1]) * 1000 for i in range(1, len(token_times))]

    return {
        "req_id": req_id,
        "ok": True,
        "status": 200,
        "ttft_ms": round(ttft * 1000, 2) if ttft is not None else None,
        "tpot_ms": round(tpot_ms, 2) if tpot_ms is not None else None,
        "e2e_ms": round(e2e * 1000, 2),
        "output_tokens": output_tokens,
        "output_chars": len(output_text),
        "itl_ms_mean": round(statistics.mean(itls), 2) if itls else None,
    }


async def run_benchmark(args):
    url = args.base_url.rstrip("/") + args.endpoint
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]

    def build_payload(i):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "stream": True,
            "max_tokens": args.max_tokens,
            # Ask the server to include usage stats in the stream when supported.
            "stream_options": {"include_usage": True},
        }
        if args.ignore_eos:
            payload["ignore_eos"] = True
        return payload

    async with aiohttp.ClientSession() as session:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} requests...", flush=True)
            for i in range(args.warmup):
                r = await send_request(session, url, build_payload(i), -i)
                if not r.get("ok"):
                    print(f"[warmup] FAILED: {r}", flush=True)
                    return 1
            print("[warmup] done", flush=True)

        sem = asyncio.Semaphore(args.concurrency)

        async def worker(req_id):
            async with sem:
                return await send_request(session, url, build_payload(req_id), req_id)

        print(
            f"[bench] url={url} n={args.num_prompts} concurrency={args.concurrency} "
            f"max_tokens={args.max_tokens} stream=True",
            flush=True,
        )
        bench_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i)) for i in range(args.num_prompts)]
        results = await asyncio.gather(*tasks)
        bench_end = time.perf_counter()

    total_s = bench_end - bench_start
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    def pct(values, p):
        if not values:
            return 0.0
        s = sorted(values)
        k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
        return s[k]

    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]
    e2es = [r["e2e_ms"] for r in ok]
    itls = [r["itl_ms_mean"] for r in ok if r.get("itl_ms_mean") is not None]
    total_out = sum((r.get("output_tokens") or 0) for r in ok)

    summary = {
        "config": {
            "url": url,
            "num_prompts": args.num_prompts,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "model": args.model,
            "stream": True,
            "ignore_eos": args.ignore_eos,
            "warmup": args.warmup,
        },
        "wall_time_s": round(total_s, 3),
        "successful": len(ok),
        "failed": len(failed),
        "requests_per_second": round(len(ok) / total_s, 3) if total_s > 0 else 0.0,
        "total_output_tokens": total_out,
        "output_tokens_per_second": round(total_out / total_s, 2) if total_s > 0 else 0.0,
        "ttft_ms": {
            "mean": round(statistics.mean(ttfts), 2) if ttfts else None,
            "median": round(statistics.median(ttfts), 2) if ttfts else None,
            "p95": round(pct(ttfts, 95), 2) if ttfts else None,
            "p99": round(pct(ttfts, 99), 2) if ttfts else None,
        },
        "tpot_ms": {
            "mean": round(statistics.mean(tpots), 2) if tpots else None,
            "median": round(statistics.median(tpots), 2) if tpots else None,
            "p95": round(pct(tpots, 95), 2) if tpots else None,
        },
        "itl_ms_mean": {
            "mean": round(statistics.mean(itls), 2) if itls else None,
            "median": round(statistics.median(itls), 2) if itls else None,
        },
        "e2e_ms": {
            "mean": round(statistics.mean(e2es), 2) if e2es else None,
            "median": round(statistics.median(e2es), 2) if e2es else None,
            "p95": round(pct(e2es, 95), 2) if e2es else None,
            "p99": round(pct(e2es, 99), 2) if e2es else None,
        },
    }
    if failed:
        summary["failure_samples"] = failed[:3]

    print(json.dumps(summary, indent=2))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps({"summary": summary, "per_request": results}, indent=2)
        )
        print(f"[out] wrote {args.output_json}", flush=True)

    return 0 if not failed else 1


def main():
    p = argparse.ArgumentParser(description="Streaming chat-omni benchmark client")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--endpoint", default="/v1/chat/completions")
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument(
        "--model",
        default="default",
        help="OpenAI 'model' field. vLLM-Omni ignores this and uses the loaded model.",
    )
    p.add_argument("--ignore-eos", action="store_true", help="Force generation up to max_tokens")
    p.add_argument("--prompts-file", help="One prompt per line")
    p.add_argument("--output-json", help="Save full results to JSON")
    args = p.parse_args()
    sys.exit(asyncio.run(run_benchmark(args)))


if __name__ == "__main__":
    main()
