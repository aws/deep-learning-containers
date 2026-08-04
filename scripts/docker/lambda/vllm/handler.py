"""Default AWS Lambda handler for the vLLM serving image.

Follows the Lambda LMI serving pattern: ONE shared vLLM OpenAI-compatible HTTP
server (`vllm serve`) is started once, and the handler is a thin proxy to it, so a
single model copy in VRAM is shared across all concurrent invocations. This is the
correct shape for a GPU serving engine under the multi-mode concurrency RIC — an
in-process `vllm.LLM` per worker would load one model copy PER worker (N copies),
which does not fit a single GPU in process/hybrid mode.

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
  MODEL_ID              HuggingFace model id or local path (default: a tiny model)
  VLLM_GPU_MEM_UTIL     gpu_memory_utilization for the server (default: 0.8)
  VLLM_TP_SIZE          tensor_parallel_size (default: visible GPU count)
  VLLM_MAX_MODEL_LEN    optional cap on the model context length
  VLLM_SERVER_PORT      port the in-container server binds (default: 8000)
  VLLM_SERVER_TIMEOUT   seconds to wait for server readiness (default: 600)
"""

import json
import os
import subprocess
import time

import requests
import torch

_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
_GPU_MEM_UTIL = os.environ.get("VLLM_GPU_MEM_UTIL", "0.8")
_MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN")
_PORT = os.environ.get("VLLM_SERVER_PORT", "8000")
_TIMEOUT = int(os.environ.get("VLLM_SERVER_TIMEOUT", "600"))
_BASE_URL = f"http://127.0.0.1:{_PORT}"

# One GPU per sandbox → device_count() is normally 1; auto-adapts if a sandbox is
# ever granted multiple GPUs. Override VLLM_TP_SIZE to pin explicitly.
_TP_SIZE = int(os.environ.get("VLLM_TP_SIZE", "0")) or max(1, torch.cuda.device_count())


def _start_server():
    """Launch the vLLM OpenAI server once and block until it is ready."""
    cmd = [
        "vllm",
        "serve",
        _MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        _PORT,
        "--gpu-memory-utilization",
        _GPU_MEM_UTIL,
        "--tensor-parallel-size",
        str(_TP_SIZE),
    ]
    if _MAX_MODEL_LEN:
        cmd += ["--max-model-len", _MAX_MODEL_LEN]
    # Detached from the handler's request loop; inherits the container env.
    subprocess.Popen(cmd)

    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{_BASE_URL}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"vLLM server did not become ready within {_TIMEOUT}s")


# Start the shared server in the mode-appropriate place. In process/hybrid the RIC
# runs @register_pre_fork once in the parent before workers spawn; in thread mode
# (and standard on-demand) pre_fork does not run, so we start at module level.
_mode = os.environ.get("AWS_LAMBDA_CONCURRENCY_MODE", "process")
_multi = bool(os.environ.get("AWS_LAMBDA_MAX_CONCURRENCY"))

if _multi and _mode in ("process", "hybrid"):
    from awslambdaric.lambda_concurrency_hooks import register_pre_fork

    register_pre_fork(_start_server)
else:
    _start_server()


def handler(event, context):
    """Proxy a single Lambda invocation to the shared vLLM OpenAI server.

    The event is the OpenAI request body, forwarded as-is. Convenience: a bare
    ``{"prompt": ...}`` is routed to /v1/completions; anything with ``messages``
    goes to /v1/chat/completions. The multi-mode RIC passes the payload as bytes,
    the standard RIC as a dict — both are normalized.
    """
    if isinstance(event, (bytes, bytearray, str)):
        event = json.loads(event or "{}")

    body = dict(event)
    body.setdefault("model", _MODEL_ID)
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"

    resp = requests.post(f"{_BASE_URL}{path}", json=body, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()
