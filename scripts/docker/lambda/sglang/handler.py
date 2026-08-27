"""Default AWS Lambda handler for the sglang serving image.

Follows the Lambda LMI serving pattern: ONE shared sglang OpenAI-compatible HTTP
server (`python -m sglang.launch_server`) is started once, and the handler is a
thin proxy to it, so a single model copy in VRAM is shared across all concurrent
invocations. An in-process `sgl.Engine` per worker would load one model copy PER
worker (N copies), which does not fit a single GPU in process/hybrid mode.

Where the server is started depends on the concurrency mode (AWS_LAMBDA_CONCURRENCY_MODE):
  - thread  (1 process × N threads): start at MODULE LEVEL — runs once in the sole
            process before threads spawn. Recommended for GPU/serving engines.
  - process / hybrid (>1 process):    start via @register_pre_fork — runs ONCE in the
            parent before workers are forked, so all workers share the one server.
            (Module-level would run in every worker → N servers colliding on the port.)
Thread mode does NOT run pre_fork hooks; process/hybrid do. Standard on-demand mode
(no AWS_LAMBDA_MAX_CONCURRENCY) also starts at module level.

Customers typically override this handler; this default lets the image serve out of
the box and gives CI a smoke target.

Environment variables:
  MODEL_ID                  HuggingFace model id, local path, or s3:// prefix (default: a tiny model)
  SGLANG_MEM_FRACTION       mem_fraction_static for the server (default: 0.8)
  SGLANG_ATTENTION_BACKEND  attention backend (default: flashinfer)
  SGLANG_TP_SIZE            tensor-parallel size (default: visible GPU count)
  SGLANG_MAX_TOTAL_TOKENS   optional cap on the KV cache token budget
  SGLANG_LOAD_FORMAT        sglang --load-format; set runai_streamer with an s3:// MODEL_ID
                            to stream safetensors from S3 to GPU (runai-model-streamer)
  SERVED_MODEL_NAME         --served-model-name; the model id clients use (set this when
                            MODEL_ID is a path/s3:// prefix so requests have a stable name)
  SGLANG_SERVER_PORT        port the in-container server binds (default: 8000)
  SGLANG_SERVER_TIMEOUT     seconds to wait for server readiness (default: 600)
"""

import json
import os
import subprocess
import time

import requests
import torch

_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
_MEM_FRACTION = os.environ.get("SGLANG_MEM_FRACTION", "0.8")
_ATTENTION_BACKEND = os.environ.get("SGLANG_ATTENTION_BACKEND", "flashinfer")
_MAX_TOTAL_TOKENS = os.environ.get("SGLANG_MAX_TOTAL_TOKENS")
# Set SGLANG_LOAD_FORMAT=runai_streamer with MODEL_ID=s3://bucket/prefix to stream
# safetensors from S3 into GPU memory via runai-model-streamer (no /tmp staging).
_LOAD_FORMAT = os.environ.get("SGLANG_LOAD_FORMAT")
# When MODEL_ID is a path/s3:// prefix, the served model name differs from it;
# set SERVED_MODEL_NAME so clients (and this handler) address a stable model id.
_SERVED_NAME = os.environ.get("SERVED_MODEL_NAME")
_PORT = os.environ.get("SGLANG_SERVER_PORT", "8000")
_TIMEOUT = int(os.environ.get("SGLANG_SERVER_TIMEOUT", "600"))
_BASE_URL = f"http://127.0.0.1:{_PORT}"

# One GPU per sandbox → device_count() is normally 1; auto-adapts if a sandbox is
# ever granted multiple GPUs. Override SGLANG_TP_SIZE to pin explicitly.
_TP_SIZE = int(os.environ.get("SGLANG_TP_SIZE", "0")) or max(1, torch.cuda.device_count())


def _start_server():
    """Launch the sglang OpenAI server once and block until it is ready."""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        _MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        _PORT,
        "--mem-fraction-static",
        _MEM_FRACTION,
        "--attention-backend",
        _ATTENTION_BACKEND,
        "--tp-size",
        str(_TP_SIZE),
    ]
    if _MAX_TOTAL_TOKENS:
        cmd += ["--max-total-tokens", _MAX_TOTAL_TOKENS]
    if _LOAD_FORMAT:
        cmd += ["--load-format", _LOAD_FORMAT]
    if _SERVED_NAME:
        cmd += ["--served-model-name", _SERVED_NAME]
    subprocess.Popen(cmd)

    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{_BASE_URL}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"sglang server did not become ready within {_TIMEOUT}s")


_mode = os.environ.get("AWS_LAMBDA_CONCURRENCY_MODE", "process")
_multi = bool(os.environ.get("AWS_LAMBDA_MAX_CONCURRENCY"))

try:
    if _multi and _mode in ("process", "hybrid"):
        # lambda_concurrency_hooks ships only in the preview RIC.
        from awslambdaric.lambda_concurrency_hooks import register_pre_fork

        register_pre_fork(_start_server)
    else:
        _start_server()
except ImportError:
    _start_server()


def handler(event, context):
    """Proxy a single Lambda invocation to the shared sglang OpenAI server.

    The event is the OpenAI request body, forwarded as-is. Convenience: a bare
    ``{"prompt": ...}`` is routed to /v1/completions; anything with ``messages``
    goes to /v1/chat/completions. The multi-mode RIC passes the payload as bytes,
    the standard RIC as a dict — both are normalized.
    """
    if isinstance(event, (bytes, bytearray, str)):
        event = json.loads(event or "{}")

    body = dict(event)
    body.setdefault("model", _SERVED_NAME or _MODEL_ID)
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"

    resp = requests.post(f"{_BASE_URL}{path}", json=body, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()
