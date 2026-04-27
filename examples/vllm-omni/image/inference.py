"""End-to-end image generation example against a local vLLM-Omni server.

Prereq: start the server with an image-generation model, e.g.
  docker run -d --gpus all -p 8080:8080 <vllm-omni-image> \
    --model black-forest-labs/FLUX.2-klein-4B
"""

import base64
import os
import pathlib

import requests

ENDPOINT = os.environ.get("OMNI_ENDPOINT", "http://localhost:8080")
OUT_PATH = pathlib.Path("out/image.png")


def generate(prompt: str, size: str = "512x512") -> bytes:
    response = requests.post(
        f"{ENDPOINT}/v1/images/generations",
        json={"prompt": prompt, "size": size, "n": 1},
        timeout=300,
    )
    response.raise_for_status()
    return base64.b64decode(response.json()["data"][0]["b64_json"])


if __name__ == "__main__":
    image = generate("a red apple on a white table, studio lighting")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(image)
    print(f"wrote {OUT_PATH} ({len(image)} bytes)")
