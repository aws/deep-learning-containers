"""End-to-end video generation example against a local vLLM-Omni server.

The /v1/videos endpoint is async — it returns a job ID immediately, and the
video is generated in the background. This script submits the job, polls
until it completes, then downloads the MP4.

Prereq: start the server with a video-generation model, e.g.
  docker run -d --gpus all -p 8080:8080 <vllm-omni-image> \
    --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers
"""

import os
import pathlib
import time

import requests

ENDPOINT = os.environ.get("OMNI_ENDPOINT", "http://localhost:8080")
OUT_PATH = pathlib.Path("out/video.mp4")
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 600


def submit_job(prompt: str) -> str:
    # /v1/videos requires multipart/form-data
    response = requests.post(
        f"{ENDPOINT}/v1/videos",
        files={
            "prompt": (None, prompt),
            "num_frames": (None, "17"),
            "num_inference_steps": (None, "4"),
            "size": (None, "480x320"),
            "seed": (None, "42"),
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["id"]


def wait_for_completion(job_id: str) -> None:
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        status = requests.get(f"{ENDPOINT}/v1/videos/{job_id}", timeout=30).json()["status"]
        if status == "succeeded":
            return
        if status == "failed":
            raise RuntimeError(f"Job {job_id} failed")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"Job {job_id} did not complete within {POLL_TIMEOUT_S}s")


def download(job_id: str, out_path: pathlib.Path) -> None:
    response = requests.get(f"{ENDPOINT}/v1/videos/{job_id}/content", timeout=60)
    response.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)


if __name__ == "__main__":
    job_id = submit_job("a dog running on a beach at sunset")
    print(f"submitted job {job_id}")
    wait_for_completion(job_id)
    download(job_id, OUT_PATH)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
