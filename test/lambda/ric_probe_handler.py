"""Diagnostic handler for the RIE and managed-instance invoke checks (not a pytest module).

Actions: echo, import_check, get_pid, gpu_procs, infer_probe.
Any other event is delegated to the image's baked serving handler.
"""

import importlib
import json
import os
import subprocess
import threading
import time

# Engine images ship a serving handler; core images do not. Absence is expected, but a
# handler that fails to import is a defect and must surface.
_baked = importlib.import_module("handler") if os.path.exists("/var/task/handler.py") else None


def _completion_ok(resp):
    """True if resp is an OpenAI-style completion with non-empty text."""
    if isinstance(resp, (str, bytes, bytearray)):
        try:
            resp = json.loads(resp)
        except ValueError:
            return False
    if not isinstance(resp, dict):
        return False
    choices = resp.get("choices") or []
    if not choices:
        return False
    first = choices[0]
    return bool(first.get("text") or (first.get("message") or {}).get("content"))


def handler(event, context):
    # Normalize: the RIC may deliver the event as bytes or as a parsed dict.
    if isinstance(event, (bytes, bytearray, str)):
        event = json.loads(event or "{}")
    action = event.get("action") if isinstance(event, dict) else None

    if action == "echo":
        return event

    if action == "import_check":
        libs = event.get("libs") or ["awslambdaric", "boto3"]
        results = {}
        for lib in libs:
            try:
                importlib.import_module(lib)
                results[lib] = True
            except Exception:
                results[lib] = False
        return results

    if action == "get_pid":
        # Sleep so concurrent invokes overlap and reveal distinct workers.
        time.sleep(event.get("sleep", 3))
        return {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "tid": threading.get_ident(),
            "request_id": context.aws_request_id,
        }

    if action == "gpu_procs":
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            pids = sorted({ln.strip() for ln in out.stdout.splitlines() if ln.strip()})
            return {"gpu_proc_count": len(pids), "gpu_pids": pids, "worker_pid": os.getpid()}
        except Exception as e:
            return {"error": repr(e)}

    if action == "infer_probe":
        if _baked is None:
            return {"ok": False, "error": "no baked serving handler on this image"}
        resp = _baked.handler(event.get("payload", {}), context)
        return {"ok": _completion_ok(resp), "pid": os.getpid(), "tid": threading.get_ident()}

    if _baked is not None:
        return _baked.handler(event, context)
    return {"error": f"unknown action: {action}"}
