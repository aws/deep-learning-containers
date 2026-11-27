"""Default AWS Lambda handler for the sglang serving image.

Follows the Lambda LMI serving pattern: ONE shared sglang OpenAI-compatible HTTP server
serves every concurrent invocation from a single model copy in VRAM, and the handler is
a thin proxy to it.

The server is owned by the engine extension (/opt/extensions/sglang-engine, source in
/opt/engine-extension/engine_extension.py), not by this module: the extension starts it
during Init, before the first invocation, and recycles the execution environment if the
server dies or stops answering /health. Replacing this handler keeps that behaviour.

Customers typically override this handler; this default lets the image serve out of
the box and gives CI a smoke target.

Environment variables (read by the extension unless noted):
  MODEL_ID                  HuggingFace model id, local path, or s3:// prefix (default: a
                            tiny model); also the request's default "model" here
  SGLANG_MEM_FRACTION       mem_fraction_static for the server (default: 0.8)
  SGLANG_ATTENTION_BACKEND  attention backend (default: flashinfer)
  SGLANG_TP_SIZE            tensor-parallel size (default: visible GPU count)
  SGLANG_MAX_TOTAL_TOKENS   optional cap on the KV cache token budget
  SGLANG_LOAD_FORMAT        sglang --load-format; set runai_streamer with an s3:// MODEL_ID
                            to stream safetensors from S3 to GPU (runai-model-streamer)
  SERVED_MODEL_NAME         --served-model-name; the model id clients use (set this when
                            MODEL_ID is a path/s3:// prefix so requests have a stable name)
  SGLANG_SERVER_ARGS        extra `sglang.launch_server` flags, shell-quoted
  SGLANG_SERVER_PORT        port the in-container server binds (default: 8000); also read here
  SGLANG_SERVER_TIMEOUT     seconds to wait for server readiness during Init (default: 600)
"""

import json
import os

import requests

_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
_SERVED_NAME = os.environ.get("SERVED_MODEL_NAME")
_BASE_URL = f"http://127.0.0.1:{os.environ.get('SGLANG_SERVER_PORT', '8000')}"
# Used only when the invocation context has no deadline (e.g. called outside the RIC).
_FALLBACK_TIMEOUT = 600


def _request_timeout(context):
    """Seconds left before the invocation deadline, less a margin to report the error."""
    remaining = getattr(context, "get_remaining_time_in_millis", None)
    if remaining is None:
        return _FALLBACK_TIMEOUT
    return max(0.1, remaining() / 1000 - 1)


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

    try:
        resp = requests.post(f"{_BASE_URL}{path}", json=body, timeout=_request_timeout(context))
    except requests.ConnectionError as e:
        raise RuntimeError(f"sglang server is not reachable at {_BASE_URL}: {e}") from e
    except requests.Timeout as e:
        raise RuntimeError(
            f"sglang server did not answer before the invocation deadline: {e}"
        ) from e
    if not resp.ok:
        raise RuntimeError(f"sglang server returned {resp.status_code}: {resp.text}")
    return resp.json()
