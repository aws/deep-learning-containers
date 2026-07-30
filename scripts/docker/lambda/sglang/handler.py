"""Default AWS Lambda handler for the sglang serving image.

The Runtime Interface Client (`python -m awslambdaric handler.handler`) imports
this module once per execution environment and then calls ``handler`` once per
invocation. We build the sglang offline engine at module scope so the model is
loaded during cold start and reused across warm invocations — sglang's HTTP
server (``sglang.launch_server``) is intentionally NOT used, since Lambda's RIC
drives a request/response loop rather than a long-lived socket.

Customers typically override this handler with their own; this default lets the
image serve out of the box and gives CI a smoke target.

Environment variables:
  MODEL_ID                  HuggingFace model id or local path (default: a tiny model)
  SGLANG_MEM_FRACTION       mem_fraction_static passed to the engine (default: 0.8)
  SGLANG_ATTENTION_BACKEND  attention backend (default: flashinfer)
  SGLANG_TP_SIZE            tensor-parallel size (default: visible GPU count)
  SGLANG_MAX_TOTAL_TOKENS   optional cap on the KV cache token budget
"""

import os

import sglang as sgl
import torch

_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
_MEM_FRACTION = float(os.environ.get("SGLANG_MEM_FRACTION", "0.8"))
_ATTENTION_BACKEND = os.environ.get("SGLANG_ATTENTION_BACKEND", "flashinfer")
_MAX_TOTAL_TOKENS = os.environ.get("SGLANG_MAX_TOTAL_TOKENS")

# Lambda binds one GPU chip per sandbox, so device_count() is normally 1; this
# auto-adapts if a sandbox is ever granted multiple GPUs. Override SGLANG_TP_SIZE
# to pin it explicitly.
_TP_SIZE = int(os.environ.get("SGLANG_TP_SIZE", "0")) or max(1, torch.cuda.device_count())

_engine_kwargs = {
    "model_path": _MODEL_ID,
    "mem_fraction_static": _MEM_FRACTION,
    "attention_backend": _ATTENTION_BACKEND,
    "tp_size": _TP_SIZE,
}
if _MAX_TOTAL_TOKENS:
    _engine_kwargs["max_total_tokens"] = int(_MAX_TOTAL_TOKENS)

# Built once at import (cold start); reused across warm invocations.
_engine = sgl.Engine(**_engine_kwargs)


def handler(event, context):
    """Generate text for a single Lambda invocation.

    Expected event shape:
        {"prompt": "...", "sampling_params": {"max_new_tokens": 128, ...}}
    ``prompt`` may be a string or a list of strings for batched generation.
    """
    prompt = event.get("prompt")
    if prompt is None:
        raise ValueError("event must include a 'prompt' field")

    sampling_params = event.get("sampling_params") or {"max_new_tokens": 128}
    outputs = _engine.generate(prompt, sampling_params)

    if isinstance(outputs, list):
        return {"outputs": [o.get("text", "") for o in outputs]}
    return {"output": outputs.get("text", "")}
