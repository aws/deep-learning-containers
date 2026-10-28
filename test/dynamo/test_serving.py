"""Serving tests for the Dynamo DLC (GPU, run inside the image).

Start the Dynamo frontend and backend worker(s) the way a user does on one node, and check
the OpenAI-compatible endpoint answers:
  aggregated:     frontend + 1 worker (GPU 0)
  disaggregated:  frontend + prefill worker (GPU 0) + decode worker (GPU 1), with the KV
                  cache moving between them (NIXL for vLLM/SGLang; TensorRT-LLM's cache
                  transceiver). A decode worker can't answer without its prefill worker, so a
                  correct response proves the KV transfer works.
Discovery uses the file backend, so no etcd/NATS is needed.

Env: DYNAMO_BACKEND (vllm | sglang | trtllm), optional DYNAMO_TEST_MODEL, DYNAMO_STARTUP_TIMEOUT,
DYNAMO_TRTLLM_CACHE_BACKEND (TensorRT-LLM's cache_transceiver_config.backend, default DEFAULT).
"""

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import pytest

BACKEND = os.environ["DYNAMO_BACKEND"]
MODEL = os.environ.get("DYNAMO_TEST_MODEL", "Qwen/Qwen3-0.6B")
STARTUP_TIMEOUT = int(os.environ.get("DYNAMO_STARTUP_TIMEOUT", "900"))
HTTP_PORT = 8000
URL = f"http://localhost:{HTTP_PORT}"
PROMPT = "What is the capital of France? Answer in one word."
TRTLLM_CACHE_BACKEND = os.environ.get("DYNAMO_TRTLLM_CACHE_BACKEND", "DEFAULT")


def trtllm_engine_args(mode):
    """TensorRT-LLM engine config for `mode` (agg | prefill | decode), after upstream's
    examples/backends/trtllm/engine_configs/qwen3/<mode>.yaml. Written as JSON, which is YAML."""
    config = {
        "tensor_parallel_size": 1,
        "max_num_tokens": 8192,
        "max_batch_size": 16,
        "trust_remote_code": True,
        "backend": "pytorch",
        "enable_chunked_prefill": True,
        "kv_cache_config": {"free_gpu_memory_fraction": 0.85},
        "cuda_graph_config": {"max_batch_size": 16},
    }
    if mode != "agg":
        config["disable_overlap_scheduler"] = mode == "prefill"
        config["cache_transceiver_config"] = {"backend": TRTLLM_CACHE_BACKEND}
    path = os.path.join(tempfile.mkdtemp(prefix="trtllm-"), f"{mode}.yaml")
    with open(path, "w") as f:
        json.dump(config, f)
    return path


def gpu_count():
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    return len([line for line in out.splitlines() if line.startswith("GPU ")])


def worker_cmd(*extra, mode="agg"):
    if BACKEND == "trtllm":
        return [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            MODEL,
            "--served-model-name",
            MODEL,
            "--extra-engine-args",
            trtllm_engine_args(mode),
            *extra,
        ]
    if BACKEND == "vllm":
        return [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            MODEL,
            "--enforce-eager",
            "--max-model-len",
            "4096",
            *extra,
        ]
    return [
        "python3",
        "-m",
        "dynamo.sglang",
        "--model-path",
        MODEL,
        "--served-model-name",
        MODEL,
        "--page-size",
        "16",
        "--tp",
        "1",
        "--trust-remote-code",
        "--disable-piecewise-cuda-graph",
        *extra,
    ]


def disagg_workers():
    """(name, env, cmd) for the prefill and decode workers, after upstream's single-node
    examples/backends/<backend>/launch/disagg.sh."""
    if BACKEND == "trtllm":
        return [
            (
                "prefill",
                {"CUDA_VISIBLE_DEVICES": "0", "DYN_SYSTEM_PORT": "8081"},
                worker_cmd("--disaggregation-mode", "prefill", mode="prefill"),
            ),
            (
                "decode",
                {"CUDA_VISIBLE_DEVICES": "1", "DYN_SYSTEM_PORT": "8082"},
                worker_cmd("--disaggregation-mode", "decode", mode="decode"),
            ),
        ]
    if BACKEND == "vllm":

        def kv(role):
            return json.dumps({"kv_connector": "NixlConnector", "kv_role": role})

        return [
            (
                "prefill",
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "DYN_SYSTEM_PORT": "8081",
                    "VLLM_NIXL_SIDE_CHANNEL_PORT": "20097",
                },
                worker_cmd(
                    "--disaggregation-mode", "prefill", "--kv-transfer-config", kv("kv_producer")
                ),
            ),
            (
                "decode",
                {"CUDA_VISIBLE_DEVICES": "1", "DYN_SYSTEM_PORT": "8082"},
                worker_cmd(
                    "--disaggregation-mode", "decode", "--kv-transfer-config", kv("kv_consumer")
                ),
            ),
        ]
    common = [
        "--disaggregation-transfer-backend",
        "nixl",
        "--disaggregation-bootstrap-port",
        "12345",
        "--host",
        "0.0.0.0",
    ]
    return [
        (
            "prefill",
            {"CUDA_VISIBLE_DEVICES": "0", "DYN_SYSTEM_PORT": "8081"},
            worker_cmd("--disaggregation-mode", "prefill", "--port", "40000", *common),
        ),
        (
            "decode",
            {"CUDA_VISIBLE_DEVICES": "1", "DYN_SYSTEM_PORT": "8082"},
            worker_cmd("--disaggregation-mode", "decode", *common),
        ),
    ]


class Deployment:
    """Processes of one local Dynamo deployment; logs go to <log_dir>/<name>.log."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.procs = {}
        self.env = dict(
            os.environ, DYN_DISCOVERY_BACKEND="file", DYN_FILE_KV=str(log_dir / "discovery")
        )

    def start(self, name, cmd, env=None):
        log = open(self.log_dir / f"{name}.log", "w")
        self.procs[name] = subprocess.Popen(
            cmd, env={**self.env, **(env or {})}, stdout=log, stderr=subprocess.STDOUT
        )

    def logs(self, lines=80):
        out = []
        for name in self.procs:
            text = (self.log_dir / f"{name}.log").read_text(errors="replace").splitlines()
            out.append(f"===== {name}.log (last {lines} lines)\n" + "\n".join(text[-lines:]))
        return "\n".join(out)

    def wait_for_model(self):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            for name, proc in self.procs.items():
                if proc.poll() is not None:
                    pytest.fail(
                        f"{name} exited with {proc.returncode} during startup\n{self.logs()}"
                    )
            try:
                with urllib.request.urlopen(f"{URL}/v1/models", timeout=5) as resp:
                    if MODEL in [m["id"] for m in json.load(resp).get("data", [])]:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(5)
        pytest.fail(f"{MODEL} not registered after {STARTUP_TIMEOUT}s\n{self.logs()}")

    def stop(self):
        for proc in self.procs.values():
            proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture
def deployment(tmp_path):
    dep = Deployment(tmp_path)
    yield dep
    dep.stop()
    print(dep.logs())  # pytest shows captured output only for failing tests


def chat(max_tokens=64):
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": PROMPT}],
        }
    ).encode()
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        assert resp.status == 200
        return json.load(resp)


def assert_serves(dep, requests=3):
    for i in range(requests):
        try:
            reply = chat()
        except urllib.error.HTTPError as e:
            pytest.fail(
                f"request {i}: HTTP {e.code}: {e.read().decode(errors='replace')}\n{dep.logs()}"
            )
        content = reply["choices"][0]["message"]["content"] or ""
        assert content.strip(), f"request {i}: empty completion: {reply}"
        assert reply["usage"]["completion_tokens"] > 0


def test_aggregated_serving(deployment):
    deployment.start(
        "frontend", ["python3", "-m", "dynamo.frontend", "--http-port", str(HTTP_PORT)]
    )
    deployment.start("worker", worker_cmd(), {"CUDA_VISIBLE_DEVICES": "0"})
    deployment.wait_for_model()
    assert_serves(deployment)


@pytest.mark.skipif(gpu_count() < 2, reason="disaggregated serving needs 2 GPUs")
def test_disaggregated_serving(deployment):
    deployment.start(
        "frontend", ["python3", "-m", "dynamo.frontend", "--http-port", str(HTTP_PORT)]
    )
    for name, env, cmd in disagg_workers():
        deployment.start(name, cmd, env)
    deployment.wait_for_model()
    assert_serves(deployment)
    if BACKEND in ("vllm", "sglang"):  # KV over NIXL; TensorRT-LLM picks its transceiver itself
        nixl = "\n".join(
            (deployment.log_dir / f"{n}.log").read_text(errors="replace")
            for n in ("prefill", "decode")
        )
        assert "was instantiated" in nixl or "KVManager initialized" in nixl, (
            "no NIXL backend in worker logs"
        )
