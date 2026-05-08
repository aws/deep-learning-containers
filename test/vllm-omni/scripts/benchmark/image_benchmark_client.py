#!/usr/bin/env python3
"""
Image generation benchmark client for vLLM-Omni /v1/images/generations.

Measures: E2E latency, throughput (images/sec), time per image step (if reported).

Usage:
    python image_benchmark_client.py \\
        --base-url http://localhost:8080 \\
        --num-prompts 20 --concurrency 2 \\
        --size 512x512 --n 1 \\
        --output-json results.json
"""

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

DEFAULT_PROMPTS = [
    "a red apple on a white table",
    "a cat sitting on a chair by a window",
    "a mountain landscape at sunset with dramatic clouds",
    "a bowl of fruit on a wooden kitchen table",
    "an old library with tall wooden shelves",
    "a futuristic city skyline at night with neon lights",
    "a cup of steaming coffee on a rainy day",
    "a forest path in autumn with fallen leaves",
    "a sailboat on calm blue water at noon",
    "a vintage bicycle leaning against a stone wall",
]


async def send_request(session, url, payload, req_id):
    start = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            text = await resp.text()
            end = time.perf_counter()
            if resp.status != 200:
                return {
                    "req_id": req_id,
                    "ok": False,
                    "status": resp.status,
                    "error": text[:300],
                    "e2e_ms": (end - start) * 1000,
                }
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                return {
                    "req_id": req_id,
                    "ok": False,
                    "status": resp.status,
                    "error": f"non-json: {text[:200]}",
                    "e2e_ms": (end - start) * 1000,
                }
            # extract images count and rough size
            data = body.get("data", [])
            n_images = len(data)
            sizes = []
            for item in data:
                b64 = item.get("b64_json")
                if b64:
                    try:
                        sizes.append(len(base64.b64decode(b64)))
                    except Exception:
                        sizes.append(None)
            return {
                "req_id": req_id,
                "ok": True,
                "status": 200,
                "e2e_ms": (end - start) * 1000,
                "n_images": n_images,
                "bytes_per_image": sizes,
            }
    except Exception as e:
        end = time.perf_counter()
        return {
            "req_id": req_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "e2e_ms": (end - start) * 1000,
        }


async def run_benchmark(args):
    url = args.base_url.rstrip("/") + args.endpoint
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]

    requests = []
    for i in range(args.num_prompts):
        payload = {
            "prompt": prompts[i % len(prompts)],
            "size": args.size,
            "n": args.n,
        }
        if args.num_inference_steps is not None:
            payload["num_inference_steps"] = args.num_inference_steps
        if args.seed is not None:
            payload["seed"] = args.seed + i  # vary seed per request
        requests.append(payload)

    async with aiohttp.ClientSession() as session:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} requests...", flush=True)
            for i in range(args.warmup):
                r = await send_request(session, url, requests[i % len(requests)], -i)
                if not r.get("ok"):
                    print(f"[warmup] FAILED: {r}", flush=True)
                    return 1
            print("[warmup] done", flush=True)

        sem = asyncio.Semaphore(args.concurrency)

        async def worker(req_id, payload):
            async with sem:
                return await send_request(session, url, payload, req_id)

        print(
            f"[bench] url={url} n={args.num_prompts} concurrency={args.concurrency} "
            f"size={args.size} steps={args.num_inference_steps}",
            flush=True,
        )
        bench_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i, payload)) for i, payload in enumerate(requests)]
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

    e2es = [r["e2e_ms"] for r in ok]
    total_images = sum(r.get("n_images", 0) for r in ok)

    summary = {
        "config": {
            "url": url,
            "num_prompts": args.num_prompts,
            "concurrency": args.concurrency,
            "size": args.size,
            "n_per_prompt": args.n,
            "num_inference_steps": args.num_inference_steps,
            "warmup": args.warmup,
        },
        "wall_time_s": round(total_s, 3),
        "successful": len(ok),
        "failed": len(failed),
        "requests_per_second": round(len(ok) / total_s, 3) if total_s > 0 else 0.0,
        "total_images": total_images,
        "images_per_second": round(total_images / total_s, 3) if total_s > 0 else 0.0,
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
        # don't save the base64 bytes
        slim = []
        for r in results:
            c = dict(r)
            c.pop("bytes_per_image", None)
            slim.append(c)
        Path(args.output_json).write_text(
            json.dumps({"summary": summary, "per_request": slim}, indent=2)
        )
        print(f"[out] wrote {args.output_json}", flush=True)

    return 0 if not failed else 1


def main():
    p = argparse.ArgumentParser(description="Image benchmark client for vLLM-Omni")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--endpoint", default="/v1/images/generations")
    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--size", default="512x512", help="Image size WIDTHxHEIGHT")
    p.add_argument("--n", type=int, default=1, help="Images per prompt")
    p.add_argument("--num-inference-steps", type=int, help="Steps override")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts-file", help="One prompt per line")
    p.add_argument("--output-json", help="Save full results to JSON")
    args = p.parse_args()
    sys.exit(asyncio.run(run_benchmark(args)))


if __name__ == "__main__":
    main()
