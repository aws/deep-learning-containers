#!/usr/bin/env python3
"""Verify RIC concurrency-mode process/thread topology on REAL AWS Lambda managed GPU.

Deploys the test-overlay image as one function per concurrency mode
(thread/process/hybrid), publishes a version, waits for it to become Active on a
GPU capacity provider, warms one execution environment, then fires MAX_CONCURRENCY
concurrent invokes and asserts:

  All images (workload = get_pid probe):
    - every concurrent invoke ran on a distinct (process, thread)
    - process count per mode: thread=1, process=MAX_CONCURRENCY, hybrid=vCPUs
    - thread mode: the single process served MAX_CONCURRENCY distinct threads

  Serving engines --engine vllm|sglang (workload = REAL inference via infer_probe):
    - every concurrent invoke returned an OpenAI completion (serving works)
    - same process/thread topology as above
    - gpu_procs == 1 in every mode (one shared engine; RIC workers proxy to it)

Uses the preview `lambda-sdk` boto3 model (register via `aws configure add-model`
before running). Cleans up every function it creates. Exit 0 = all pass.
"""

import argparse
import concurrent.futures as cf
import json
import sys
import time

import boto3

MODEL_PROMPT = {"prompt": "The capital of France is", "max_tokens": 16}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image", required=True, help="Test-overlay image URI (same acct+region as function)"
    )
    p.add_argument("--region", required=True)
    p.add_argument("--capacity-provider-arn", required=True)
    p.add_argument("--execution-role-arn", required=True)
    p.add_argument("--engine", choices=["none", "vllm", "sglang"], default="none")
    p.add_argument("--modes", default="thread,process,hybrid")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--memory-size", type=int, default=16384)  # 16 GiB @ 4:1 -> 4 vCPU
    p.add_argument("--ephemeral-size", type=int, default=10240)
    p.add_argument("--accelerator-memory", type=int, default=24)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--name-prefix", default="rictest")
    return p.parse_args()


class Runner:
    def __init__(self, a):
        self.a = a
        self.lam = boto3.client("lambda-sdk", region_name=a.region)
        self.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
        self.created = []

    # ---- lifecycle helpers -------------------------------------------------
    def _cfg(self, name, q=None):
        kw = {"FunctionName": name}
        if q:
            kw["Qualifier"] = q
        c = self.lam.get_function_configuration20150331v2(**kw)
        return c.get("State"), c.get("LastUpdateStatus")

    def _wait_ready(self, name, timeout=1200):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                st, lu = self._cfg(name)
            except self.lam.exceptions.ResourceNotFoundException:
                return False
            if st in ("Active", "ActiveNonInvocable") and lu in (None, "Successful"):
                return True
            if st == "Failed":
                return False
            time.sleep(15)
        return False

    def _wait_active(self, name, q, timeout=1500):
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout:
            st, _ = self._cfg(name, q)
            if st != last:
                print(f"    [{name}:{q}] {st} ({int(time.time() - t0)}s)", flush=True)
                last = st
            if st == "Active":
                return True
            if st == "Failed":
                sr = self.lam.get_function_configuration20150331v2(
                    FunctionName=name, Qualifier=q
                ).get("StateReason")
                print(f"    !! Failed: {sr}", flush=True)
                return False
            time.sleep(20)
        return False

    def _delete(self, name):
        try:
            self.lam.get_function_configuration20150331v2(FunctionName=name)
            self._wait_ready(name)
            self.lam.delete_function20150331(FunctionName=name)
            for _ in range(60):
                try:
                    self.lam.get_function_configuration20150331v2(FunctionName=name)
                    time.sleep(5)
                except self.lam.exceptions.ResourceNotFoundException:
                    return
        except self.lam.exceptions.ResourceNotFoundException:
            pass

    def _env(self, mode):
        a = self.a
        env = {
            "AWS_LAMBDA_CONCURRENCY_MODE": mode,
            "AWS_LAMBDA_MAX_CONCURRENCY": str(a.max_concurrency),
        }
        if a.engine != "none":
            env["MODEL_ID"] = a.model_id
            env["HF_HOME"] = "/tmp/huggingface"
            if a.engine == "vllm":
                env.update({"VLLM_GPU_MEM_UTIL": "0.4", "VLLM_MAX_MODEL_LEN": "2048"})
            else:
                env.update({"SGLANG_MEM_FRACTION": "0.4", "SGLANG_MAX_TOTAL_TOKENS": "2048"})
        return env

    def _deploy(self, mode):
        a = self.a
        name = f"{a.name_prefix}-{a.engine}-{mode}"
        self._delete(name)
        self.created.append(name)
        self.lam.create_function20150331(
            FunctionName=name,
            PackageType="Image",
            Code={"ImageUri": a.image},
            Role=a.execution_role_arn,
            Timeout=600,
            MemorySize=a.memory_size,
            EphemeralStorage={"Size": a.ephemeral_size},
            Environment={"Variables": self._env(mode)},
            LoggingConfig={"LogFormat": "JSON", "LogGroup": f"/aws/lambda/{name}"},
            CapacityProviderConfig={
                "LambdaManagedInstancesCapacityProviderConfig": {
                    "CapacityProviderArn": a.capacity_provider_arn,
                    "ExecutionEnvironmentMemoryGiBPerVCpu": 4,
                    "PerExecutionEnvironmentMaxConcurrency": a.max_concurrency,
                }
            },
            AcceleratorConfig={"AcceleratorMemorySize": a.accelerator_memory},
        )
        if not self._wait_ready(name):
            raise RuntimeError(f"{name} $LATEST not ready")
        v = self.lam.publish_version20150331(FunctionName=name)["Version"]
        print(f"    published v{v}, warming...", flush=True)
        if not self._wait_active(name, v):
            raise RuntimeError(f"{name}:{v} did not reach Active")
        return name, v

    # ---- invocation helpers ------------------------------------------------
    def _invoke(self, name, q, payload):
        r = self.lam.invoke20150331(
            FunctionName=f"{name}:{q}", Payload=json.dumps(payload).encode()
        )
        return r.get("StatusCode"), r.get("FunctionError"), r["Payload"].read().decode()

    def _concurrent(self, name, q, payload, n):
        with cf.ThreadPoolExecutor(max_workers=n) as ex:
            return [
                f.result() for f in [ex.submit(self._invoke, name, q, payload) for _ in range(n)]
            ]

    # ---- per-mode test -----------------------------------------------------
    def test_mode(self, mode):
        a = self.a
        n = a.max_concurrency
        print(f"\n=== {a.engine} / {mode} ===", flush=True)
        name, v = self._deploy(mode)
        res = {"mode": mode}

        # Warm exactly one execution environment before the burst so all concurrent
        # invokes land on it (PerEEMaxConcurrency == max_concurrency keeps them there),
        # making the proc/thread counts reflect the mode and not EE scale-out.
        self._invoke(name, v, {"action": "get_pid", "sleep": 0})

        # Burst of n concurrent invokes. Engines run REAL inference (infer_probe);
        # others run a get_pid probe. Both report the worker's (pid, tid), so the
        # burst reveals process/thread topology while doing the actual workload.
        if a.engine != "none":
            payload = {"action": "infer_probe", "payload": MODEL_PROMPT}
        else:
            payload = {"action": "get_pid", "sleep": 3}

        expected_procs = {"thread": 1, "process": n, "hybrid": a.memory_size // 1024 // 4}.get(mode)
        pairs, ok = self._burst(name, v, payload, n)
        procs = len({p for p, _ in pairs})
        # A stray 2nd execution environment (scheduler scale-out) would inflate the
        # process count. Retry the burst once before trusting a mismatch — the EEs are
        # warm now, so the retry consolidates onto one. A real regression still fails.
        if len(pairs) != n or procs != expected_procs:
            pairs, ok = self._burst(name, v, payload, n)

        complete = len(pairs) == n
        procs = len({p for p, _ in pairs}) if complete else None
        res["workload_ok"] = ok and complete
        res["procs"] = procs  # distinct worker processes
        res["handlers"] = (
            len(set(pairs)) if complete else None
        )  # distinct (proc, thread) = concurrent handlers
        res["threads_in_proc"] = len({t for _, t in pairs}) if complete and procs == 1 else None

        if a.engine != "none":
            sc, fe, g = self._invoke(name, v, {"action": "gpu_procs"})
            res["gpu_procs"] = json.loads(g).get("gpu_proc_count") if sc == 200 and not fe else None

        print(f"    {res}", flush=True)
        self._delete(name)
        return res

    def _burst(self, name, v, payload, n):
        """Fire n concurrent invokes; return (list of (pid,tid), all_ok)."""
        outs = self._concurrent(name, v, payload, n)
        pairs, ok = [], True
        for sc, fe, b in outs:
            if sc != 200 or fe:
                ok = False
                continue
            d = json.loads(b)
            if "ok" in d:
                ok = ok and bool(d.get("ok"))  # engine: real completion returned
            if d.get("pid") is not None and d.get("tid") is not None:
                pairs.append((d["pid"], d["tid"]))
        return pairs, ok

    def run(self):
        results = {}
        try:
            for m in self.modes:
                try:
                    results[m] = self.test_mode(m)
                except Exception as e:
                    print(f"    !! {m} errored: {e}", flush=True)
                    results[m] = {"mode": m, "error": str(e)}
        finally:
            for name in self.created:
                self._delete(name)
        return results


def evaluate(a, results):
    mc = a.max_concurrency
    expected_hybrid = a.memory_size // 1024 // 4  # K == vCPUs (memory / 4 GiB @ 4:1)
    checks = {}
    # Workload succeeded and every concurrent invoke ran on a distinct (proc, thread).
    for m, r in results.items():
        checks[f"{m}:workload_ok"] = r.get("workload_ok") is True
        checks[f"{m}:{mc}_handlers"] = r.get("handlers") == mc
    # Per-mode process/thread topology.
    if "thread" in results:
        t = results["thread"]
        checks[f"thread:1proc_{mc}threads"] = t.get("procs") == 1 and t.get("threads_in_proc") == mc
    if "process" in results:
        checks["process:Nproc"] = results["process"].get("procs") == mc
    if "hybrid" in results:
        checks[f"hybrid:{expected_hybrid}proc"] = results["hybrid"].get("procs") == expected_hybrid
    # Engines: real inference already asserted via workload_ok; also one shared GPU process.
    if a.engine != "none":
        for m, r in results.items():
            checks[f"{m}:one_gpu_proc"] = r.get("gpu_procs") == 1
    return checks


def main():
    a = parse_args()
    r = Runner(a)
    results = r.run()
    checks = evaluate(a, results)
    print("\n===================== SUMMARY =====================", flush=True)
    print(f"engine={a.engine} image={a.image}")
    print("procs=" + json.dumps({m: results[m].get("procs") for m in results}))
    print("handlers=" + json.dumps({m: results[m].get("handlers") for m in results}))
    print(
        "threads_in_proc(thread mode)="
        + json.dumps({m: results[m].get("threads_in_proc") for m in results})
    )
    if a.engine != "none":
        print("gpu_procs=" + json.dumps({m: results[m].get("gpu_procs") for m in results}))
    ok = True
    for k, v in sorted(checks.items()):
        ok &= bool(v)
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("==================================================")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
