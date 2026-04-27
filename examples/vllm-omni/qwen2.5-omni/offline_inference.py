"""Offline inference against a local vLLM-Omni server running Qwen2.5-Omni-3B.

Assumes the server is already running on http://localhost:8080 (see the
Qwen2.5-Omni tutorial for `docker run` instructions).
"""

import base64
import json
import os
import pathlib

import requests

ENDPOINT = os.environ.get("OMNI_ENDPOINT", "http://localhost:8080")
MODEL = "Qwen/Qwen2.5-Omni-3B"
SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

# Three per-stage sampling params (thinker, talker, code2wav) are REQUIRED for
# clean audio. The built-in defaults produce noise. Do not omit.
SAMPLING_PARAMS_LIST = [
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": 42,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
    {
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40,
        "max_tokens": 2048,
        "seed": 42,
        "detokenize": True,
        "repetition_penalty": 1.05,
        "stop_token_ids": [8294],
    },
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 2048,
        "seed": 42,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
]


def generate_audio(prompt: str, out_path: pathlib.Path) -> None:
    payload = {
        "model": MODEL,
        "modalities": ["audio"],
        "sampling_params_list": SAMPLING_PARAMS_LIST,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
    }
    response = requests.post(
        f"{ENDPOINT}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=600,
    )
    response.raise_for_status()
    audio_b64 = response.json()["choices"][0]["message"]["audio"]["data"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(audio_b64))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate_audio(
        "Tell me a short, calming bedtime lullaby story for a 6-year-old girl.",
        pathlib.Path("out/lullaby.wav"),
    )
