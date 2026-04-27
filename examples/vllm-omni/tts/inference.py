"""End-to-end TTS example against a local vLLM-Omni server.

Prereq: start the server with a TTS model, e.g.
  docker run -d --gpus all -p 8080:8080 <vllm-omni-image> \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
"""

import os
import pathlib

import requests

ENDPOINT = os.environ.get("OMNI_ENDPOINT", "http://localhost:8080")
OUT_PATH = pathlib.Path("out/speech.wav")


def synthesize(text: str, voice: str = "vivian", language: str = "English") -> bytes:
    response = requests.post(
        f"{ENDPOINT}/v1/audio/speech",
        json={"input": text, "voice": voice, "language": language},
        timeout=300,
    )
    response.raise_for_status()
    return response.content


if __name__ == "__main__":
    audio = synthesize("Hello from vLLM-Omni. This is a text to speech demo.")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(audio)
    print(f"wrote {OUT_PATH} ({len(audio)} bytes)")
