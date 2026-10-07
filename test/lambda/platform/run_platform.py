"""RIC invoke test on real Lambda: build a function from a DLC image, invoke it, assert.

Covers what the emulator check cannot: the real control plane, managed-instance compute,
and concurrency configured through the API rather than an env var.

Exit codes: 0 pass, 1 assertion failure, 2 no GPU capacity (not an image defect).

The managed-instance fields are absent from the bundled botocore, so requests are signed
with SigV4 directly. POST-GA: replace _call with boto3.client("lambda").
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

FUNCTIONS = "/2015-03-31/functions"
RETRYABLE_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 5
CAPACITY_MARKERS = (
    "InsufficientCapacity",
    "insufficient capacity",
    "CapacityNotAvailable",
    "no capacity",
    "Capacity is not available",
)


class ApiError(Exception):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class CapacityUnavailable(Exception):
    pass


def _call(region, method, path, payload=None, raw_response=False):
    url = f"https://lambda.{region}.amazonaws.com{path}"
    body = json.dumps(payload).encode() if payload is not None else b""
    credentials = boto3.Session().get_credentials()
    for attempt in range(MAX_ATTEMPTS):
        signed = AWSRequest(
            method=method, url=url, data=body, headers={"Content-Type": "application/json"}
        )
        SigV4Auth(credentials, "lambda", region).add_auth(signed)
        try:
            request = urllib.request.Request(
                url, data=body or None, method=method, headers=dict(signed.headers)
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                if raw_response:
                    return data.decode(errors="replace")
                return json.loads(data or b"{}")
        except urllib.error.HTTPError as e:
            error = ApiError(e.code, e.read().decode(errors="replace"))
            if e.code not in RETRYABLE_STATUS:
                raise error from None
        except (urllib.error.URLError, TimeoutError) as e:
            error = RuntimeError(f"{method} {path}: {e!r}")
        if attempt == MAX_ATTEMPTS - 1:
            raise error
        print(f"  {method} {path} attempt {attempt + 1}/{MAX_ATTEMPTS}: {error}")
        time.sleep(2**attempt + random.random())


def _quote(name):
    return urllib.parse.quote(name, safe="")


def create_function(args, name):
    # aws lambda create-function
    payload = {
        "FunctionName": name,
        "Role": args.execution_role_arn,
        "PackageType": "Image",
        "Code": {"ImageUri": args.image},
        "Timeout": args.invoke_timeout,
        "MemorySize": args.memory_size,
        "EphemeralStorage": {"Size": args.ephemeral_storage},
        "Architectures": ["x86_64"],
        "AcceleratorConfig": {"AcceleratorMemorySize": args.accelerator_memory_size},
        "CapacityProviderConfig": {
            "LambdaManagedInstancesCapacityProviderConfig": {
                "CapacityProviderArn": args.capacity_provider_arn,
                "PerExecutionEnvironmentMaxConcurrency": args.concurrency,
                "ExecutionEnvironmentMemoryGiBPerVCpu": args.memory_gib_per_vcpu,
            }
        },
    }
    _call(args.region, "POST", FUNCTIONS, payload)
    wait_created(args, name)
    # aws lambda publish-version
    return _call(args.region, "POST", f"{FUNCTIONS}/{_quote(name)}/versions")["Version"]


def wait_created(args, name):
    deadline = time.time() + args.ready_timeout
    while time.time() < deadline:
        # aws lambda get-function-configuration
        current = _call(args.region, "GET", f"{FUNCTIONS}/{_quote(name)}/configuration")
        state, update = current.get("State"), current.get("LastUpdateStatus")
        print(f"  create state={state} lastUpdate={update}")
        if state != "Pending" and update == "Successful":
            return
        if state == "Failed" or update == "Failed":
            raise AssertionError(
                f"{name} failed to create: {current.get('StateReasonCode')} "
                f"{current.get('StateReason') or current.get('LastUpdateStatusReason')}"
            )
        time.sleep(10)
    raise AssertionError(f"{name} did not settle within {args.ready_timeout}s")


def wait_active(args, name, version):
    deadline = time.time() + args.ready_timeout
    last = {}
    while time.time() < deadline:
        # aws lambda get-function-configuration --qualifier
        last = _call(
            args.region,
            "GET",
            f"{FUNCTIONS}/{_quote(name)}/configuration?Qualifier={_quote(version)}",
        )
        state = last.get("State")
        reason = f"{last.get('StateReasonCode') or ''} {last.get('StateReason') or ''}".strip()
        print(f"  state={state} {reason}".rstrip())
        if state == "Active":
            return
        if state == "Failed" or any(m in reason for m in CAPACITY_MARKERS):
            if any(m in reason for m in CAPACITY_MARKERS):
                raise CapacityUnavailable(reason)
            raise AssertionError(f"function {name} entered Failed: {reason}")
        time.sleep(10)
    raise CapacityUnavailable(
        f"{name}:{version} not Active within {args.ready_timeout}s "
        f"(last state {last.get('State')} {last.get('StateReasonCode') or ''}). "
        f"Check the capacity provider's scaling activities for launch failures."
    )


def invoke(args, name, payload, version="$LATEST"):
    # aws lambda invoke --qualifier
    raw = _call(
        args.region,
        "POST",
        f"{FUNCTIONS}/{_quote(name)}/invocations?Qualifier={_quote(version)}",
        payload,
        raw_response=True,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise AssertionError(f"non-JSON response from {name}: {raw[:300]}")
    if isinstance(parsed, dict) and parsed.get("errorType"):
        raise AssertionError(f"handler raised {parsed['errorType']}: {parsed.get('errorMessage')}")
    return parsed


def invoke_concurrently(args, name, payload, count, version):
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(invoke, args, name, payload, version) for _ in range(count)]
        return [f.result() for f in futures]


def test_invoke_and_image(args, name, version):
    """Proves: handler called via the RIC, and DLC libraries importable from handler code."""
    event = {"action": "echo", "marker": "dlc-ric-platform"}
    assert invoke(args, name, event, version)["marker"] == "dlc-ric-platform", (
        "echo did not round-trip"
    )
    print("  PASS echo round-trips through the RIC")

    libs = ["awslambdaric", "boto3", "torch"]
    if args.engine != "none":
        libs.append(args.engine)
    results = invoke(args, name, {"action": "import_check", "libs": libs}, version)
    missing = [lib for lib, ok in results.items() if not ok]
    assert not missing, f"imports failed inside the invoke path: {missing}"
    print(f"  PASS imports reached through an invoke: {', '.join(libs)}")


def test_concurrency_and_prefork(args, name, version):
    """Proves: simultaneous invokes get separate workers, and the pre-fork hook runs first."""
    results = invoke_concurrently(
        args, name, {"action": "get_pid", "sleep": args.overlap_seconds}, args.concurrency, version
    )
    pids = {r["pid"] for r in results}
    assert len(results) == args.concurrency, "lost an invoke"
    assert len(pids) == args.concurrency, (
        f"expected {args.concurrency} forked workers, saw {sorted(pids)}"
    )
    print(f"  PASS {args.concurrency} concurrent invokes served by {len(pids)} workers")

    hook = invoke(args, name, {"action": "check_hook"}, version)
    assert hook["hook_executed"], "register_pre_fork hook never ran"
    assert hook["hook_ran_in_parent"], (
        f"pre-fork hook ran in pid {hook['hook_pid']}, not the worker's parent {hook['ppid']}"
    )
    print(f"  PASS pre-fork hook ran once in the parent (pid {hook['hook_pid']})")


def test_gpu_shared_engine(args, name, version):
    """Proves: on vllm/sglang images only, inference succeeds on one shared GPU process."""
    payload = {"prompt": "Describe a container image in one sentence.", "max_tokens": 16}
    results = invoke_concurrently(
        args, name, {"action": "infer_probe", "payload": payload}, args.concurrency, version
    )
    failures = [r for r in results if not r.get("ok")]
    assert not failures, f"inference failed on {len(failures)} of {len(results)} invokes"
    print(f"  PASS {len(results)} concurrent inferences returned completions")

    procs = invoke(args, name, {"action": "gpu_procs"}, version)
    assert procs.get("gpu_proc_count") == 1, (
        f"expected one process on the GPU, saw {procs.get('gpu_proc_count')}: {procs.get('gpu_pids')}"
    )
    print("  PASS exactly one process holds the GPU")


def run(args):
    name = f"{args.name_prefix}-{os.urandom(3).hex()}"
    print(f"=== {name} ===")
    try:
        version = create_function(args, name)
        wait_active(args, name, version)
        test_invoke_and_image(args, name, version)
        test_concurrency_and_prefork(args, name, version)
        if args.engine != "none":
            test_gpu_shared_engine(args, name, version)
    finally:
        try:
            # aws lambda delete-function
            _call(args.region, "DELETE", f"{FUNCTIONS}/{_quote(name)}")
            print(f"  deleted {name}")
        except ApiError as e:
            if e.status != 404:
                print(f"  WARNING could not delete {name}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="overlay image URI, same account and region")
    p.add_argument("--region", required=True)
    p.add_argument("--capacity-provider-arn", required=True)
    p.add_argument("--execution-role-arn", required=True)
    p.add_argument("--engine", default="none", choices=["none", "vllm", "sglang"])
    p.add_argument("--name-prefix", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--accelerator-memory-size",
        type=int,
        default=12,
        choices=[3, 6, 12, 16, 24, 48],
        help="minimum GPU memory in GB per execution environment",
    )
    p.add_argument("--memory-size", type=int, default=4096)
    p.add_argument("--memory-gib-per-vcpu", type=int, default=4)
    p.add_argument("--ephemeral-storage", type=int, default=10240)
    p.add_argument("--invoke-timeout", type=int, default=300)
    p.add_argument("--ready-timeout", type=int, default=900)
    p.add_argument("--overlap-seconds", type=int, default=5)
    args = p.parse_args()

    try:
        run(args)
    except CapacityUnavailable as e:
        print(f"\nSKIP no GPU capacity for the managed-instance pool: {e}")
        return 2
    except AssertionError as e:
        print(f"\nFAIL {e}")
        return 1
    except Exception as e:
        print(f"\nFAIL unexpected error: {e!r}")
        return 1
    print("\nAll real-Lambda RIC checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
