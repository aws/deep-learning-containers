"""Unified RIC-topology test handler for real Lambda managed-GPU tests.

Layered onto any Lambda preview image (base/cupy/pytorch/vllm/sglang) via
Dockerfile.test with CMD ["test_handler.handler"]. It exposes diagnostic actions
used by run_topology.py and, for serving engines, delegates real inference to the
image's baked handler (imported as `base`).

Actions (JSON event `{"action": ...}`):
  echo          -> returns the event verbatim
  import_check  -> {lib: bool} for a fixed set of libraries
  get_pid       -> {pid, request_id} after a short sleep (overlaps concurrent invokes)
  check_hook    -> {hook_executed} — whether the pre-fork marker was written
  gpu_procs     -> {gpu_proc_count, gpu_pids} from nvidia-smi compute-apps
  <else>        -> delegated to the baked serving handler (engines); error otherwise
"""

import importlib
import os
import subprocess
import threading
import time

# Baked serving handler (present only on vllm/sglang images). Importing it runs the
# image's module-level setup — including its own @register_pre_fork server start.
try:
    import handler as base
except Exception:
    base = None

# Register our own pre-fork marker hook so check_hook works on every image. Fires
# once in the parent before workers spawn in process/hybrid modes (not thread).
try:
    from awslambdaric.lambda_concurrency_hooks import register_pre_fork

    @register_pre_fork
    def _write_prefork_marker():
        with open("/tmp/prefork_marker", "w") as f:
            f.write("hook_ran")
except Exception:
    pass

_IMPORT_LIBS = ["awslambdaric", "boto3", "torch", "transformers", "cupy", "vllm", "sglang"]


def _valid_completion(resp):
    """True if resp is an OpenAI-style completion with non-empty text/content."""
    if isinstance(resp, (str, bytes)):
        try:
            import json as _json

            resp = _json.loads(resp)
        except Exception:
            return False
    if not isinstance(resp, dict):
        return False
    choices = resp.get("choices")
    if not choices:
        return False
    c0 = choices[0]
    return bool(c0.get("text") or (c0.get("message") or {}).get("content"))


def handler(event, context):
    action = event.get("action") if isinstance(event, dict) else None

    if action == "echo":
        return event

    if action == "import_check":
        results = {}
        for lib in _IMPORT_LIBS:
            try:
                importlib.import_module(lib)
                results[lib] = True
            except Exception:
                results[lib] = False
        return results

    if action == "get_pid":
        # sleep so concurrent invokes overlap in-flight -> reveal true proc/thread topology
        time.sleep(event.get("sleep", 3))
        return {
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "request_id": context.aws_request_id,
        }

    if action == "infer_probe":
        # Run REAL inference (engines) and report the worker's pid/tid alongside, so a
        # burst of concurrent real inferences reveals worker topology in one pass.
        if base is None:
            return {"ok": False, "error": "no baked serving handler on this image"}
        resp = base.handler(event.get("payload", {}), context)
        ok = _valid_completion(resp)
        return {"ok": ok, "pid": os.getpid(), "tid": threading.get_ident()}

    if action == "check_hook":
        return {"hook_executed": os.path.exists("/tmp/prefork_marker"), "pid": os.getpid()}

    if action == "gpu_procs":
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            pids = sorted({line.strip() for line in out.stdout.splitlines() if line.strip()})
            return {"gpu_proc_count": len(pids), "gpu_pids": pids, "worker_pid": os.getpid()}
        except Exception as e:
            return {"error": repr(e)}

    # No diagnostic action -> real inference for serving engines; error otherwise.
    if base is not None:
        return base.handler(event, context)
    return {"error": f"unknown action: {action}"}
