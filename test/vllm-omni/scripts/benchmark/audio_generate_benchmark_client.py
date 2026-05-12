#!/usr/bin/env python3
"""
Audio-generate benchmark client for vLLM-Omni /v1/audio/generate endpoint.

Endpoint introduced in vllm-omni v0.20.0 (vllm-project/vllm-omni#1794) for
diffusion-based audio models like stable-audio-open. Returns a single binary
WAV blob (no streaming).

Measures: TTFB, E2E latency, output audio duration, real-time factor (RTF),
request throughput, and audio-throughput (seconds-of-audio per wall-second).
Same metric set as tts_benchmark_client.py so threshold validators
(`min_rps`, `min_audio_rtf_mult`, `max_p95_e2e_ms`) work unchanged.

Usage:
    python audio_generate_benchmark_client.py \\
        --base-url http://localhost:8080 \\
        --num-prompts 8 --concurrency 1 \\
        --audio-length 5.0 --num-inference-steps 50 \\
        --output-json results.json
"""

import argparse
import asyncio
import json
import statistics
import struct
import sys
import time
from pathlib import Path

import aiohttp

DEFAULT_PROMPTS = [
    "A dog barking in a quiet park",
    "Rainfall on a tin roof",
    "A jazz piano improvisation",
    "Ocean waves crashing on a rocky shore",
    "A cat purring softly",
    "Gentle wind through pine trees",
    "Footsteps on a wooden floor",
    "A distant thunderstorm approaching",
]


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Parse WAV header to extract audio duration. Supports PCM 16-bit mono/stereo."""
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return 0.0
    i = 12
    sample_rate = 0
    num_channels = 0
    bits_per_sample = 0
    data_size = 0
    while i + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[i : i + 4]
        chunk_size = struct.unpack("<I", wav_bytes[i + 4 : i + 8])[0]
        if chunk_id == b"fmt ":
            num_channels = struct.unpack("<H", wav_bytes[i + 10 : i + 12])[0]
            sample_rate = struct.unpack("<I", wav_bytes[i + 12 : i + 16])[0]
            bits_per_sample = struct.unpack("<H", wav_bytes[i + 22 : i + 24])[0]
        elif chunk_id == b"data":
            data_size = chunk_size
            break
        i += 8 + chunk_size
    if not (sample_rate and num_channels and bits_per_sample and data_size):
        return 0.0
    bytes_per_sample = bits_per_sample // 8
    total_samples = data_size // (num_channels * bytes_per_sample)
    return total_samples / sample_rate


async def send_request(session, url, payload, req_id):
    """Send one /v1/audio/generate request and return timing metrics."""
    start = time.perf_counter()
    ttfb = None
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            first_chunk = await resp.content.readany()
            ttfb = time.perf_counter() - start
            body = bytearray(first_chunk)
            async for chunk in resp.content.iter_any():
                body.extend(chunk)
            end = time.perf_counter()
            if resp.status != 200:
                return {
                    "req_id": req_id,
                    "ok": False,
                    "status": resp.status,
                    "error": bytes(body[:200]).decode("utf-8", errors="replace"),
                    "e2e_ms": (end - start) * 1000,
                    "ttfb_ms": ttfb * 1000,
                }
            duration_s = wav_duration_seconds(bytes(body))
            e2e = end - start
            return {
                "req_id": req_id,
                "ok": True,
                "status": 200,
                "bytes": len(body),
                "audio_duration_s": duration_s,
                "ttfb_ms": ttfb * 1000,
                "e2e_ms": e2e * 1000,
                "rtf": (e2e / duration_s) if duration_s > 0 else None,
            }
    except Exception as e:
        end = time.perf_counter()
        return {
            "req_id": req_id,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "e2e_ms": (end - start) * 1000,
            "ttfb_ms": (ttfb * 1000) if ttfb else None,
        }


async def run_benchmark(args):
    url = args.base_url.rstrip("/") + args.endpoint
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [p.strip() for p in Path(args.prompts_file).read_text().splitlines() if p.strip()]
    if not prompts:
        raise ValueError("no prompts")

    def build_payload(i):
        payload = {
            "input": prompts[i % len(prompts)],
            "audio_length": args.audio_length,
            "response_format": args.response_format,
        }
        if args.audio_start is not None:
            payload["audio_start"] = args.audio_start
        if args.guidance_scale is not None:
            payload["guidance_scale"] = args.guidance_scale
        if args.num_inference_steps is not None:
            payload["num_inference_steps"] = args.num_inference_steps
        if args.seed is not None:
            payload["seed"] = args.seed
        if args.negative_prompt:
            payload["negative_prompt"] = args.negative_prompt
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
            f"audio_length={args.audio_length} steps={args.num_inference_steps}",
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

    ttfbs = [r["ttfb_ms"] for r in ok if r.get("ttfb_ms") is not None]
    e2es = [r["e2e_ms"] for r in ok]
    durations = [r["audio_duration_s"] for r in ok]
    rtfs = [r["rtf"] for r in ok if r.get("rtf") is not None]

    summary = {
        "config": {
            "url": url,
            "num_prompts": args.num_prompts,
            "concurrency": args.concurrency,
            "audio_length": args.audio_length,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "response_format": args.response_format,
            "warmup": args.warmup,
        },
        "wall_time_s": round(total_s, 3),
        "successful": len(ok),
        "failed": len(failed),
        "requests_per_second": round(len(ok) / total_s, 3) if total_s > 0 else 0.0,
        "total_audio_duration_s": round(sum(durations), 3),
        "audio_throughput_s_per_s": round(sum(durations) / total_s, 3) if total_s > 0 else 0.0,
        "ttfb_ms": {
            "mean": round(statistics.mean(ttfbs), 2) if ttfbs else None,
            "median": round(statistics.median(ttfbs), 2) if ttfbs else None,
            "p95": round(pct(ttfbs, 95), 2) if ttfbs else None,
            "p99": round(pct(ttfbs, 99), 2) if ttfbs else None,
        },
        "e2e_ms": {
            "mean": round(statistics.mean(e2es), 2) if e2es else None,
            "median": round(statistics.median(e2es), 2) if e2es else None,
            "p95": round(pct(e2es, 95), 2) if e2es else None,
            "p99": round(pct(e2es, 99), 2) if e2es else None,
        },
        "rtf": {
            "mean": round(statistics.mean(rtfs), 4) if rtfs else None,
            "median": round(statistics.median(rtfs), 4) if rtfs else None,
            "p95": round(pct(rtfs, 95), 4) if rtfs else None,
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
    p = argparse.ArgumentParser(description="Audio-generate benchmark client for vLLM-Omni")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--endpoint", default="/v1/audio/generate")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument(
        "--audio-length",
        type=float,
        default=5.0,
        help="Output duration in seconds (~47s max for stable-audio-open-1.0)",
    )
    p.add_argument("--audio-start", type=float, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument(
        "--response-format", default="wav", choices=["wav", "mp3", "flac", "pcm", "aac", "opus"]
    )
    p.add_argument("--prompts-file", help="One prompt per line")
    p.add_argument("--output-json", help="Save full results to JSON")
    args = p.parse_args()
    sys.exit(asyncio.run(run_benchmark(args)))


if __name__ == "__main__":
    main()
