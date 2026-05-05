#!/usr/bin/env python3
"""
Video generation benchmark client for vLLM-Omni /v1/videos endpoint.

The /v1/videos endpoint is asynchronous:
1. POST /v1/videos (multipart/form-data) → returns {id, status: "queued"}
2. GET /v1/videos/{id} → poll until status in ("completed", "failed")
3. GET /v1/videos/{id}/content → download the MP4

Measures: submission latency, server-reported inference time, wall-clock E2E,
request throughput (videos/sec), video file size.

Usage:
    python video_benchmark_client.py \\
        --base-url http://localhost:8080 \\
        --num-prompts 5 --concurrency 1 \\
        --num-frames 17 --size 480x320 --num-inference-steps 4 \\
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
    "a dog running on a beach",
    "a cat chasing a ball in a sunny garden",
    "a timelapse of clouds moving across a blue sky",
    "a car driving down a winding mountain road",
    "waves crashing against rocks at sunset",
    "a person walking through a busy city street at night",
    "a flower blooming in fast motion",
    "a bird flying across a forest canopy",
    "a hot air balloon drifting over green hills",
    "a train passing through a snowy landscape",
]


async def submit(session, url, form_fields, req_id):
    """Submit a video generation request. Returns (created_at, submit_latency_ms, body)."""
    start = time.perf_counter()
    form = aiohttp.FormData()
    for k, v in form_fields.items():
        form.add_field(k, str(v))
    async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=60)) as resp:
        body = await resp.json()
        end = time.perf_counter()
        if resp.status != 200:
            raise RuntimeError(f"submit failed: status={resp.status} body={body}")
        return start, (end - start) * 1000, body


async def poll_until_done(session, base_url, video_id, poll_interval_s=1.5, max_wait_s=900):
    """Poll the video status endpoint until completed or failed."""
    deadline = time.perf_counter() + max_wait_s
    last_progress = -1
    while time.perf_counter() < deadline:
        async with session.get(
            f"{base_url}/v1/videos/{video_id}", timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            body = await resp.json()
            status = body.get("status", "?")
            progress = body.get("progress", 0)
            if progress != last_progress:
                last_progress = progress
            if status == "completed":
                return body
            if status == "failed":
                raise RuntimeError(f"video {video_id} failed: {body.get('error')}")
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(f"video {video_id} not done in {max_wait_s}s")


async def fetch_content(session, base_url, video_id):
    """Fetch the final MP4 bytes."""
    async with session.get(
        f"{base_url}/v1/videos/{video_id}/content", timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        if resp.status != 200:
            return None
        return await resp.read()


async def run_one(
    session, base_url, submit_url, form_fields, req_id, poll_interval_s, download_content
):
    """End-to-end: submit → poll → (optionally) download content."""
    t0 = time.perf_counter()
    try:
        start_ts, submit_ms, submit_body = await submit(session, submit_url, form_fields, req_id)
        video_id = submit_body.get("id")
        final = await poll_until_done(session, base_url, video_id, poll_interval_s=poll_interval_s)
        content_bytes = 0
        if download_content:
            content = await fetch_content(session, base_url, video_id)
            content_bytes = len(content) if content else 0
        end = time.perf_counter()
        return {
            "req_id": req_id,
            "ok": True,
            "video_id": video_id,
            "submit_ms": round(submit_ms, 2),
            "e2e_ms": round((end - t0) * 1000, 2),
            "inference_time_s": final.get("inference_time_s"),
            "progress": final.get("progress"),
            "video_bytes": content_bytes,
        }
    except Exception as e:
        end = time.perf_counter()
        return {
            "req_id": req_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "e2e_ms": round((end - t0) * 1000, 2),
        }


async def run_benchmark(args):
    base_url = args.base_url.rstrip("/")
    submit_url = base_url + args.endpoint
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]

    base_form = {
        "num_frames": args.num_frames,
        "size": args.size,
        "num_inference_steps": args.num_inference_steps,
    }

    requests = []
    for i in range(args.num_prompts):
        form = dict(base_form)
        form["prompt"] = prompts[i % len(prompts)]
        if args.seed is not None:
            form["seed"] = args.seed + i
        requests.append(form)

    async with aiohttp.ClientSession() as session:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} requests...", flush=True)
            for i in range(args.warmup):
                r = await run_one(
                    session,
                    base_url,
                    submit_url,
                    requests[i % len(requests)],
                    -i,
                    args.poll_interval_s,
                    download_content=False,
                )
                if not r.get("ok"):
                    print(f"[warmup] FAILED: {r}", flush=True)
                    return 1
            print("[warmup] done", flush=True)

        sem = asyncio.Semaphore(args.concurrency)

        async def worker(req_id, form):
            async with sem:
                return await run_one(
                    session,
                    base_url,
                    submit_url,
                    form,
                    req_id,
                    args.poll_interval_s,
                    args.download_content,
                )

        print(
            f"[bench] url={submit_url} n={args.num_prompts} concurrency={args.concurrency} "
            f"size={args.size} frames={args.num_frames} steps={args.num_inference_steps}",
            flush=True,
        )
        bench_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i, form)) for i, form in enumerate(requests)]
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
    submits = [r["submit_ms"] for r in ok]
    infer_s = [r["inference_time_s"] for r in ok if r.get("inference_time_s") is not None]

    summary = {
        "config": {
            "url": submit_url,
            "num_prompts": args.num_prompts,
            "concurrency": args.concurrency,
            "num_frames": args.num_frames,
            "size": args.size,
            "num_inference_steps": args.num_inference_steps,
            "download_content": args.download_content,
            "warmup": args.warmup,
            "poll_interval_s": args.poll_interval_s,
        },
        "wall_time_s": round(total_s, 3),
        "successful": len(ok),
        "failed": len(failed),
        "requests_per_second": round(len(ok) / total_s, 3) if total_s > 0 else 0.0,
        "videos_per_second": round(len(ok) / total_s, 3) if total_s > 0 else 0.0,
        "submit_ms": {
            "mean": round(statistics.mean(submits), 2) if submits else None,
            "median": round(statistics.median(submits), 2) if submits else None,
            "p95": round(pct(submits, 95), 2) if submits else None,
        },
        "e2e_ms": {
            "mean": round(statistics.mean(e2es), 2) if e2es else None,
            "median": round(statistics.median(e2es), 2) if e2es else None,
            "p95": round(pct(e2es, 95), 2) if e2es else None,
            "p99": round(pct(e2es, 99), 2) if e2es else None,
        },
        "server_inference_time_s": {
            "mean": round(statistics.mean(infer_s), 3) if infer_s else None,
            "median": round(statistics.median(infer_s), 3) if infer_s else None,
            "p95": round(pct(infer_s, 95), 3) if infer_s else None,
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
    p = argparse.ArgumentParser(description="Video benchmark client for vLLM-Omni")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--endpoint", default="/v1/videos")
    p.add_argument("--num-prompts", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--num-frames", type=int, default=17)
    p.add_argument("--size", default="480x320")
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--poll-interval-s", type=float, default=1.5)
    p.add_argument(
        "--download-content",
        action="store_true",
        help="Also GET /content to measure download time (default: skip)",
    )
    p.add_argument("--prompts-file", help="One prompt per line")
    p.add_argument("--output-json", help="Save full results to JSON")
    args = p.parse_args()
    sys.exit(asyncio.run(run_benchmark(args)))


if __name__ == "__main__":
    main()
