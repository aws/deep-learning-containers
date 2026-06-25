#!/usr/bin/env python3
"""
TTS benchmark client for vLLM-Omni /v1/audio/speech endpoint.

Measures: TTFB, E2E latency, output audio duration, real-time factor (RTF),
and request throughput under configurable concurrency.

Usage:
    python tts_benchmark_client.py \\
        --base-url http://localhost:8080 \\
        --num-prompts 50 --concurrency 4 \\
        --voice vivian --language English \\
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
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning has transformed the way we build software.",
    "The weather is unusually warm for this time of year.",
    "Could you help me understand how neural networks work?",
    "Thank you for your patience while we resolve this issue.",
    "Please remember to lock the door before leaving.",
    "I would like a coffee with a little bit of milk and sugar.",
    "Artificial intelligence is changing every industry rapidly.",
    "The train to London departs from platform three in five minutes.",
]


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Parse WAV header to extract audio duration. Supports PCM 16-bit mono/stereo."""
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return 0.0
    # Walk chunks to find 'fmt ' and 'data'
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
    """Send one TTS request and return timing metrics."""
    start = time.perf_counter()
    ttfb = None
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            # TTFB = time until we read the first byte of the body
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
                    "error": body[:200].decode("utf-8", errors="replace"),
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

    # Load reference audio for Base task-type (voice cloning)
    ref_audio_uri = None
    if args.task_type == "Base":
        if not args.ref_audio or not args.ref_text:
            raise ValueError("--task-type Base requires --ref-audio and --ref-text")
        ref_bytes = Path(args.ref_audio).read_bytes()
        import base64 as _b64

        ref_audio_uri = "data:audio/wav;base64," + _b64.b64encode(ref_bytes).decode("ascii")

    # Build the request queue
    requests = []
    for i in range(args.num_prompts):
        payload = {
            "input": prompts[i % len(prompts)],
            "language": args.language,
        }
        if args.task_type == "Base":
            payload["task_type"] = "Base"
            payload["ref_audio"] = ref_audio_uri
            payload["ref_text"] = args.ref_text
        else:
            payload["voice"] = args.voice
        requests.append(payload)

    # Warmup
    async with aiohttp.ClientSession() as session:
        if args.warmup > 0:
            print(f"[warmup] {args.warmup} requests...", flush=True)
            for i in range(args.warmup):
                await send_request(session, url, requests[i % len(requests)], -i)
            print("[warmup] done", flush=True)

        # Concurrent benchmark
        sem = asyncio.Semaphore(args.concurrency)

        async def worker(req_id, payload):
            async with sem:
                return await send_request(session, url, payload, req_id)

        print(
            f"[bench] url={url} n={args.num_prompts} concurrency={args.concurrency} "
            f"voice={args.voice} language={args.language}",
            flush=True,
        )
        bench_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(i, payload)) for i, payload in enumerate(requests)]
        results = await asyncio.gather(*tasks)
        bench_end = time.perf_counter()

    # Aggregate
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
            "task_type": args.task_type,
            "voice": args.voice if args.task_type == "CustomVoice" else None,
            "ref_audio": args.ref_audio if args.task_type == "Base" else None,
            "language": args.language,
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
    p = argparse.ArgumentParser(description="TTS benchmark client for vLLM-Omni")
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--endpoint", default="/v1/audio/speech")
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--voice", default="vivian")
    p.add_argument("--language", default="English")
    p.add_argument(
        "--task-type",
        choices=["CustomVoice", "Base"],
        default="CustomVoice",
        help="TTS task type. Base requires --ref-audio and --ref-text for voice cloning.",
    )
    p.add_argument("--ref-audio", help="Path to reference WAV file (required for --task-type Base)")
    p.add_argument(
        "--ref-text", help="Transcript of reference audio (required for --task-type Base)"
    )
    p.add_argument("--prompts-file", help="One prompt per line")
    p.add_argument("--output-json", help="Save full results to JSON")
    args = p.parse_args()
    sys.exit(asyncio.run(run_benchmark(args)))


if __name__ == "__main__":
    main()
