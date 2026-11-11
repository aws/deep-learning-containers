#!/usr/bin/env python3
"""Disaggregated serving test of a Dynamo image on SageMaker HyperPod (EKS) with the HyperPod Inference add-on.

Runs on the CI runner, not inside the image. It:

1. waits its turn (one test at a time: the GPU groups hold one node per worker), then scales the prefill and decode
   instance groups up to at least one node each,
2. deploys a DynamoGraphDeployment in its own namespace: a frontend, and a prefill and a decode worker pinned to
   different nodes, so the KV cache moves between nodes over NIXL (LIBFABRIC on EFA images, UCX otherwise),
3. sends chat requests through the frontend and checks the answers, the workers' restarts and the NIXL backend,
4. saves logs, deletes the namespace, and scales the GPU groups back to the counts they had before CI scaled them up
   (normally 0), unless another test is waiting.

While it runs, a background thread heartbeats its namespace and renews the lock (a Lease), so a holder that dies
(cancelled or timed-out job) is detected within LEASE_STALE_S: the next test takes the lock over and deletes the dead
holder's namespace, and a dead test's namespace never keeps the GPU groups up.

With --fallback-groups, a pair of groups whose nodes don't come up within PAIR_NODE_TIMEOUT_S (no on-demand capacity in
that AZ) is scaled back and the fallback pair is tried; later tests in the same queue start with the pair that worked.

Usage: run_hyperpod_test.py --image-uri URI --runtime-version 1.6.2 --backend vllm --efa-flavor efa
    --eks-cluster NAME --hyperpod-cluster NAME --prefill-group GROUP --decode-group GROUP
    [--fallback-groups PREFILL,DECODE] [--region us-west-2] [--log-dir DIR] [--model Qwen/Qwen3-0.6B]
(--help for details)
"""

import argparse
import calendar
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"
CI_NAMESPACE_LABEL = "dynamo-ci=hyperpod"
NODE_TIMEOUT_S = 45 * 60  # on-demand GPU capacity can be slow to arrive
# Nodes normally join in 3-6 min; a pair that has none after this lacks capacity, so try the fallback pair (which then
# gets the rest of NODE_TIMEOUT_S).
PAIR_NODE_TIMEOUT_S = 15 * 60
READY_TIMEOUT_S = 30 * 60  # first pull of a ~15 GB image, then the model download and load
REQUESTS = 3
# One test at a time: a Lease in kube-system whose creation is the lock. The holder renews it every HEARTBEAT_S; one not
# renewed for LEASE_STALE_S belongs to a dead holder and is taken over.
LOCK = ["-n", "kube-system", "lease/dynamo-ci-hyperpod-lock"]
# The GPU groups' counts before CI scaled them up, restored by the last test out (created by the first test in), and
# the group pair that last got nodes, which later tests try first.
BASELINE = ["-n", "kube-system", "configmap/dynamo-ci-hyperpod-baseline"]
HEARTBEAT_ANNOTATION = "dynamo-ci/heartbeat"
HEARTBEAT_S = 60
LEASE_STALE_S = 5 * 60
# Budget against the workflow's 300-minute job timeout: lock wait 180 + after the lock at most ~105 (nodes 45, pods 30,
# model 10, requests ~6, KV check 2, namespace delete 10) + setup and log upload.
LOCK_WAIT_S = 3 * 60 * 60


def configure(argv=None):
    """Parse the flags into the module's settings (also used to render the DGD without a cluster)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image-uri", required=True, help="image to test")
    parser.add_argument(
        "--runtime-version",
        required=True,
        help="the image's Dynamo version (metadata.framework_version)",
    )
    parser.add_argument(
        "--backend", required=True, choices=["vllm", "sglang", "trtllm"], help="metadata.backend"
    )
    parser.add_argument(
        "--efa-flavor", required=True, choices=["efa", "noefa"], help="build.efa_flavor"
    )
    parser.add_argument("--eks-cluster", required=True, help="EKS cluster name")
    parser.add_argument("--hyperpod-cluster", required=True, help="HyperPod cluster name")
    parser.add_argument(
        "--prefill-group", required=True, help="instance group for the prefill worker"
    )
    parser.add_argument("--decode-group", required=True,
                        help="instance group for the decode worker (in the prefill group's subnet, for EFA)")  # fmt: skip
    parser.add_argument("--fallback-groups", default="",
                        help="PREFILL,DECODE groups to use when the main pair gets no nodes (e.g. another AZ)")  # fmt: skip
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--log-dir", default="hyperpod-test-logs")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args(argv)
    global \
        IMAGE, \
        RUNTIME_VERSION, \
        BACKEND, \
        EFA, \
        EKS_CLUSTER, \
        HYPERPOD_CLUSTER, \
        PREFILL_GROUP, \
        DECODE_GROUP
    global REGION, LOG_DIR, MODEL, NAMESPACE, DGD, PAIRS
    IMAGE, RUNTIME_VERSION, BACKEND, EFA = (
        args.image_uri,
        args.runtime_version,
        args.backend,
        args.efa_flavor == "efa",
    )
    EKS_CLUSTER, HYPERPOD_CLUSTER = args.eks_cluster, args.hyperpod_cluster
    PREFILL_GROUP, DECODE_GROUP = args.prefill_group, args.decode_group
    PAIRS = [(PREFILL_GROUP, DECODE_GROUP)]
    if args.fallback_groups:
        fallback = tuple(g.strip() for g in args.fallback_groups.split(","))
        if len(fallback) != 2 or not all(fallback):
            parser.error("--fallback-groups takes PREFILL,DECODE")
        PAIRS.append(fallback)
    REGION, LOG_DIR, MODEL = args.region, Path(args.log_dir), args.model
    run_id = (
        f"{os.environ.get('GITHUB_RUN_ID', 'local')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    )
    NAMESPACE = f"dynamo-ci-{run_id}-{BACKEND}-{args.efa_flavor}"
    DGD = f"dlc-{BACKEND}"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(*args, check=True, capture=True, stdin=None):
    result = subprocess.run(args, input=stdin, text=True, capture_output=capture)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout if capture else ""


def kubectl(*args, **kw):
    return run("kubectl", *args, **kw)


def aws(*args):
    return json.loads(run("aws", "--region", REGION, "--output", "json", *args) or "{}")


# ===== DynamoGraphDeployment =====


def worker_container(mode, group):
    """The worker's command, environment and resources for BACKEND and EFA."""
    nixl_backend = "LIBFABRIC" if EFA else "UCX"
    environment = [{"name": "NIXL_LOG_LEVEL", "value": "INFO"}]
    if EFA:
        environment.append({"name": "FI_PROVIDER", "value": "efa"})
    if BACKEND == "vllm":
        kv = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"backends": [nixl_backend]},
        }
        command = ["python3", "-m", "dynamo.vllm"]
        args = [
            "--model",
            MODEL,
            "--max-model-len",
            "4096",
            "--disaggregation-mode",
            mode,
            "--kv-transfer-config",
            json.dumps(kv),
        ]
    elif BACKEND == "sglang":
        environment.append({"name": "SGLANG_DISAGGREGATION_NIXL_BACKEND", "value": nixl_backend})
        command = ["python3", "-m", "dynamo.sglang"]
        args = ["--model-path", MODEL, "--served-model-name", MODEL, "--page-size", "16", "--tp", "1",
                "--trust-remote-code", "--disaggregation-mode", mode, "--disaggregation-transfer-backend", "nixl",
                "--disaggregation-bootstrap-port", "12345", "--host", "0.0.0.0"]  # fmt: skip
    else:
        # Engine settings follow upstream's examples/backends/trtllm/engine_configs/qwen3/{prefill,decode}.yaml.
        # On EFA, TensorRT-LLM's cache transceiver uses NIXL with the image's LIBFABRIC plugin; otherwise its default.
        engine = {
            "tensor_parallel_size": 1, "max_num_tokens": 8192, "trust_remote_code": True, "backend": "pytorch",
            "enable_chunked_prefill": True, "disable_overlap_scheduler": mode == "prefill",
            "cuda_graph_config": {"max_batch_size": 16}, "kv_cache_config": {"free_gpu_memory_fraction": 0.85},
            "cache_transceiver_config": {"backend": "NIXL" if EFA else "DEFAULT"},
        }  # fmt: skip
        if EFA:
            environment.append({"name": "TRTLLM_NIXL_KVCACHE_BACKEND", "value": "LIBFABRIC"})
        command = ["sh", "-c"]
        args = [f"echo '{json.dumps(engine)}' > /tmp/engine.yaml && exec python3 -m dynamo.trtllm "
                f"--model-path {MODEL} --served-model-name {MODEL} --extra-engine-args /tmp/engine.yaml "
                f"--disaggregation-mode {mode}"]  # fmt: skip
    limits = {"nvidia.com/gpu": "1"}
    if EFA:
        limits["vpc.amazonaws.com/efa"] = "1"
    return {
        "nodeSelector": {GROUP_LABEL: group},
        "containers": [
            {
                "name": "main",
                "image": IMAGE,
                # CI tags (pr-NUMBER) move between commits and the GPU nodes are reused: always pull.
                "imagePullPolicy": "Always",
                "command": command,
                "args": args,
                "env": environment,
                "resources": {"limits": limits},
            }  # fmt: skip
        ],
    }


def dgd_manifest():
    # The operator takes the Dynamo version from a semver image tag; CI tags aren't semver, so set it.
    def component(name, kind, spec):
        return {"name": name, "type": kind, "replicas": 1, "runtimeVersionOverride": RUNTIME_VERSION,
                "podTemplate": {"spec": spec}}  # fmt: skip

    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": DGD, "namespace": NAMESPACE},
        "spec": {
            "components": [
                component(
                    "Frontend",
                    "frontend",
                    {"containers": [{"name": "main", "image": IMAGE, "imagePullPolicy": "Always"}]},
                ),
                component("prefill", "prefill", worker_container("prefill", PREFILL_GROUP)),
                component("decode", "decode", worker_container("decode", DECODE_GROUP)),
            ]
        },
    }


# ===== Instance groups =====


def instance_groups():
    return {
        g["InstanceGroupName"]: g
        for g in aws("sagemaker", "describe-cluster", "--cluster-name", HYPERPOD_CLUSTER)[
            "InstanceGroups"
        ]
    }


def wanted_counts():
    """Nodes each GPU group needs: one per worker (two if prefill and decode share a group)."""
    counts = {}
    for group in (PREFILL_GROUP, DECODE_GROUP):
        counts[group] = counts.get(group, 0) + 1
    return counts


def set_counts(counts):
    # UpdateCluster takes each group's full spec: resend every setting the CI cluster's groups use (including the
    # weekly AMI update schedule), or the call clears it. Add a key here when the cluster's groups get a new setting.
    keys = ("InstanceGroupName", "InstanceType", "LifeCycleConfig", "ExecutionRole", "ThreadsPerCore",
            "InstanceStorageConfigs", "OverrideVpcConfig", "ScheduledUpdateConfig")  # fmt: skip
    groups = instance_groups()
    # UpdateCluster rejects a call that changes nothing (a group kept running at its count, e.g. on reserved
    # capacity), and a group recorded earlier may no longer exist: send only real changes.
    missing = sorted(name for name in counts if name not in groups)
    if missing:
        log(f"instance groups not in the cluster, skipped: {', '.join(missing)}")
    counts = {
        name: n
        for name, n in counts.items()
        if name in groups and groups[name].get("TargetCount") != n
    }
    if not counts:
        log("instance groups already at the requested counts")
        return
    update = [
        {**{k: groups[name][k] for k in keys if k in groups[name]}, "InstanceCount": n}
        for name, n in counts.items()
    ]
    log(f"instance groups -> {counts}")
    run("aws", "--region", REGION, "sagemaker", "update-cluster", "--cluster-name", HYPERPOD_CLUSTER,
        "--instance-groups", json.dumps(update))  # fmt: skip


def state():
    """The shared state in BASELINE: {"counts": {group: count before CI}, "active": "PREFILL,DECODE" or ""}."""
    data = json.loads(
        kubectl("get", *BASELINE, "-o", "jsonpath={.data}", check=False).strip() or "{}"
    )
    return {
        "counts": json.loads(data["counts"]) if "counts" in data else None,
        "active": data.get("active", ""),
    }


def save_state(counts, active):
    manifest = {"apiVersion": "v1", "kind": "ConfigMap",
                "metadata": {"name": BASELINE[2].split("/")[1], "namespace": "kube-system"},
                "data": {"counts": json.dumps(counts), "active": active}}  # fmt: skip
    kubectl("apply", "-f", "-", stdin=json.dumps(manifest))


def baseline():
    return state()["counts"]


def scale_up():
    """Scale the groups up to what the workers need. The first test in records the groups' counts before CI touched
    them, so the last test out restores them instead of scaling to 0 (holding the lock, so no one else is scaling)."""
    groups = instance_groups()
    current = state()
    recorded = current["counts"]
    missing = {
        g: groups[g]["TargetCount"]
        for g in wanted_counts()
        if recorded is None or g not in recorded
    }
    if missing:
        counts = {**(recorded or {}), **missing}
        save_state(counts, current["active"])
        log(f"baseline counts: {counts}")
    short = {name: n for name, n in wanted_counts().items() if groups[name]["TargetCount"] < n}
    if short:
        set_counts(short)


def use_pair(pair):
    global PREFILL_GROUP, DECODE_GROUP
    PREFILL_GROUP, DECODE_GROUP = pair


def release_other_pairs(pair):
    """Put every other pair's groups back to their baseline counts: the instance quota is per account, so a pair left
    scaled up (e.g. for a queued test) would block this one."""
    recorded = baseline() or {}
    groups = instance_groups()
    others = {g: recorded[g] for other in PAIRS if other != pair for g in other
              if g not in pair and g in recorded and groups[g]["TargetCount"] != recorded[g]}  # fmt: skip
    if others:
        set_counts(others)


def get_gpu_nodes():
    """Scale a group pair up and wait for its nodes; on a pair that gets none in time, try the next. The pair being
    tried is recorded, so a queued test starts with the pair whose groups are already up (or that last worked)."""
    active = state()["active"]
    pairs = sorted(PAIRS, key=lambda pair: ",".join(pair) != active)
    for i, pair in enumerate(pairs):
        use_pair(pair)
        last = i == len(pairs) - 1
        timeout = NODE_TIMEOUT_S - PAIR_NODE_TIMEOUT_S * i if last else PAIR_NODE_TIMEOUT_S
        log(f"GPU groups: prefill {PREFILL_GROUP}, decode {DECODE_GROUP}")
        release_other_pairs(pair)
        scale_up()
        current = state()
        if current["active"] != ",".join(pair):
            save_state(current["counts"] or {}, ",".join(pair))
        try:
            wait_for("GPU nodes", gpu_nodes_ready, timeout, interval_s=30)
            return
        except TimeoutError:
            if last:
                raise
            log(
                f"no GPU nodes in {PREFILL_GROUP}/{DECODE_GROUP} after {timeout} s; trying the next pair"
            )


def gpu_nodes_ready():
    """True when every worker has a Ready node in its group with a free GPU (and EFA device) advertised."""
    nodes = json.loads(kubectl("get", "nodes", "-o", "json"))["items"]
    ready = {}
    for node in nodes:
        group = node["metadata"]["labels"].get(GROUP_LABEL)
        conditions = {c["type"]: c["status"] for c in node["status"].get("conditions", [])}
        allocatable = node["status"].get("allocatable", {})
        has_devices = int(allocatable.get("nvidia.com/gpu", "0")) >= 1 and (
            not EFA or int(allocatable.get("vpc.amazonaws.com/efa", "0")) >= 1
        )
        if conditions.get("Ready") == "True" and has_devices:
            ready[group] = ready.get(group, 0) + 1
    return all(ready.get(group, 0) >= n for group, n in wanted_counts().items())


def now_micro():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def age_s(stamp):
    """Seconds since a Kubernetes timestamp (2026-11-03T01:02:03Z or with microseconds)."""
    return time.time() - calendar.timegm(
        time.strptime(stamp.split(".")[0].rstrip("Z"), "%Y-%m-%dT%H:%M:%S")
    )


HOLDING_LOCK = False
NAMESPACE_DELETED = threading.Event()


def acquire_lock():
    name = LOCK[2].split("/")[1]

    def take():
        global HOLDING_LOCK
        lease = {"apiVersion": "coordination.k8s.io/v1", "kind": "Lease",
                 "metadata": {"name": name, "namespace": "kube-system"},
                 "spec": {"holderIdentity": NAMESPACE, "leaseDurationSeconds": LEASE_STALE_S,
                          "acquireTime": now_micro(), "renewTime": now_micro()}}  # fmt: skip
        if run("kubectl", "create", "-f", "-", stdin=json.dumps(lease), check=False, capture=True):
            HOLDING_LOCK = True
            return True
        current = json.loads(kubectl("get", *LOCK, "-o", "json"))
        spec = current["spec"]
        if (
            age_s(spec.get("renewTime") or current["metadata"]["creationTimestamp"])
            <= LEASE_STALE_S
        ):
            return False
        # A dead holder: take the lease over. replace carries the resourceVersion, so only one waiter wins.
        holder = spec.get("holderIdentity", "")
        log(f"taking over the lock from {holder} (not renewed since {spec.get('renewTime')})")
        current["spec"] = lease["spec"]
        if not run(
            "kubectl", "replace", "-f", "-", stdin=json.dumps(current), check=False, capture=True
        ):
            return False
        HOLDING_LOCK = True
        if holder.startswith("dynamo-ci-") and holder != NAMESPACE:
            kubectl(
                "delete", "namespace", holder, "--ignore-not-found", "--wait=false", check=False
            )
        return True

    log("waiting for the HyperPod test lock")
    wait_for("the HyperPod test lock", take, LOCK_WAIT_S, interval_s=30)
    log("lock acquired")


def release_lock():
    global HOLDING_LOCK
    holder = kubectl("get", *LOCK, "-o", "jsonpath={.spec.holderIdentity}", check=False).strip()
    if holder == NAMESPACE:
        kubectl("delete", *LOCK, check=False)
    HOLDING_LOCK = False


def heartbeat(stop):
    """Until stop is set: mark this test's namespace alive and renew the lock while holding it."""
    while not stop.wait(HEARTBEAT_S):
        try:
            if not NAMESPACE_DELETED.is_set():
                kubectl("annotate", "namespace", NAMESPACE, "--overwrite",
                        f"{HEARTBEAT_ANNOTATION}={int(time.time())}")  # fmt: skip
            if HOLDING_LOCK:
                kubectl(
                    "patch",
                    *LOCK,
                    "--type",
                    "merge",
                    "-p",
                    json.dumps({"spec": {"renewTime": now_micro()}}),
                )
        except RuntimeError as e:
            log(f"heartbeat: {e}")


def live_other_tests():
    """Other CI test namespaces that are alive (heartbeat within LEASE_STALE_S). Dead ones are deleted."""
    live = []
    for ns in json.loads(kubectl("get", "namespaces", "-l", CI_NAMESPACE_LABEL, "-o", "json"))[
        "items"
    ]:
        name = ns["metadata"]["name"]
        if name == NAMESPACE or ns.get("status", {}).get("phase") == "Terminating":
            continue
        beat = ns["metadata"].get("annotations", {}).get(HEARTBEAT_ANNOTATION)
        idle = time.time() - int(beat) if beat else age_s(ns["metadata"]["creationTimestamp"])
        if idle > LEASE_STALE_S:
            log(f"deleting {name}: no heartbeat for {int(idle)} s")
            kubectl("delete", "namespace", name, "--wait=false", check=False)
        else:
            live.append(name)
    return live


def scale_down_if_idle():
    """Restore the GPU groups' baseline counts unless another live CI test exists (a test creates its namespace
    before it waits for the lock, so the last test out restores them). Called while holding the lock."""
    others = live_other_tests()
    if others:
        log(f"leaving the GPU groups up for {len(others)} other test(s): {' '.join(others)}")
        return
    recorded = baseline()
    if recorded is None:
        return
    set_counts(recorded)
    kubectl("delete", *BASELINE, "--ignore-not-found", check=False)


# ===== Test =====


def wait_for(description, predicate, timeout_s, interval_s=15):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if predicate():
                return
        except (RuntimeError, ValueError, KeyError, OSError, urllib.error.URLError) as e:
            log(f"waiting for {description}: {e}")
        time.sleep(interval_s)
    raise TimeoutError(f"{description} not ready after {timeout_s} s")


def pods():
    return json.loads(kubectl("get", "pods", "-n", NAMESPACE, "-o", "json"))["items"]


def pod_role(pod):
    name = pod["metadata"]["name"]
    return next(
        (role for role in ("frontend", "prefill", "decode") if role in name.lower()), "other"
    )


def all_pods_ready():
    ready = {
        pod_role(p)
        for p in pods()
        if all(c.get("ready") for c in p["status"].get("containerStatuses", [{}]))
    }
    return {"frontend", "prefill", "decode"} <= ready


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(url, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def send_requests(port):
    base = f"http://127.0.0.1:{port}"

    def model_listed():
        _, models = http(f"{base}/v1/models", timeout=10)
        return any(m.get("id") == MODEL for m in models.get("data", []))

    wait_for("the frontend to list the model", model_listed, 10 * 60)
    for i in range(1, REQUESTS + 1):
        start = time.time()
        status, answer = http(
            f"{base}/v1/chat/completions",
            {
                "model": MODEL,
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France? Answer in one sentence.",
                    }
                ],
            },  # fmt: skip
        )
        content = answer["choices"][0]["message"]["content"]
        log(f"request {i}: HTTP {status} in {time.time() - start:.2f} s: {content[-120:]!r}")
        if status != 200 or not content.strip():
            raise AssertionError(f"request {i}: HTTP {status}, empty answer")


def logs_of(role):
    pod = next((p["metadata"]["name"] for p in pods() if pod_role(p) == role), None)
    return kubectl("logs", "-n", NAMESPACE, pod, check=False) if pod else ""


def check_workers():
    for pod in pods():
        restarts = sum(c.get("restartCount", 0) for c in pod["status"].get("containerStatuses", []))
        if pod_role(pod) in ("prefill", "decode") and restarts:
            raise AssertionError(f"{pod['metadata']['name']} restarted {restarts} time(s)")
    # TensorRT-LLM without EFA uses its own transceiver, not NIXL.
    if BACKEND == "trtllm" and not EFA:
        return
    nixl_backend = "LIBFABRIC" if EFA else "UCX"
    # vLLM and SGLang create NIXL agents through its Python API; TensorRT-LLM through its own C++ transfer agent.
    if BACKEND == "trtllm":
        expected = f"using NIXL backend: {nixl_backend}"
    else:
        expected = f"Backend {nixl_backend} was instantiated"
    for role in ("prefill", "decode"):
        if expected not in logs_of(role):
            raise AssertionError(f"{role}: no '{expected}' in its log")
    log(f"both workers: {expected}")
    if BACKEND == "vllm":
        # vLLM logs its KV transfer metrics periodically.
        def transferred():
            lines = [
                line
                for line in logs_of("decode").splitlines()
                if "Num successful transfers=" in line
            ]
            return lines and int(lines[-1].split("Num successful transfers=")[1].split(",")[0]) > 0

        wait_for("a successful KV transfer in the decode log", transferred, 120)
        log("decode: KV transfers succeeded")


def save_logs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "dgd.yaml").write_text(json.dumps(dgd_manifest(), indent=2))
    for name, args in {
        "pods.txt": ["get", "pods", "-n", NAMESPACE, "-o", "wide"],
        "describe-pods.txt": ["describe", "pods", "-n", NAMESPACE],
        "events.txt": ["get", "events", "-n", NAMESPACE, "--sort-by=.lastTimestamp"],
        "dgd.txt": ["get", "dynamographdeployment", DGD, "-n", NAMESPACE, "-o", "yaml"],
        "nodes.txt": ["get", "nodes", "-o", "wide", "-L", GROUP_LABEL],
    }.items():
        (LOG_DIR / name).write_text(kubectl(*args, check=False))
    for pod in pods():
        name = pod["metadata"]["name"]
        (LOG_DIR / f"{name}.log").write_text(kubectl("logs", "-n", NAMESPACE, name, check=False))
        (LOG_DIR / f"{name}.previous.log").write_text(
            kubectl("logs", "-n", NAMESPACE, name, "--previous", check=False)
        )


def main():
    os.environ["KUBECONFIG"] = os.path.join(tempfile.mkdtemp(), "kubeconfig")
    run("aws", "eks", "update-kubeconfig", "--name", EKS_CLUSTER, "--region", REGION)
    log(
        f"{IMAGE}: {BACKEND}, {'EFA' if EFA else 'no EFA'}; prefill on {PREFILL_GROUP}, decode on {DECODE_GROUP}"
    )
    port_forward = None
    interrupted = False
    kubectl("create", "namespace", NAMESPACE)
    kubectl("label", "namespace", NAMESPACE, CI_NAMESPACE_LABEL)
    kubectl("annotate", "namespace", NAMESPACE, f"{HEARTBEAT_ANNOTATION}={int(time.time())}")
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    try:
        acquire_lock()
        get_gpu_nodes()
        log("GPU nodes ready")
        kubectl("apply", "-f", "-", stdin=json.dumps(dgd_manifest()))
        wait_for(
            "frontend, prefill and decode pods", all_pods_ready, READY_TIMEOUT_S, interval_s=20
        )
        log("pods ready")
        port = free_port()
        port_forward = subprocess.Popen(
            ["kubectl", "port-forward", "-n", NAMESPACE, f"svc/{DGD}-frontend", f"{port}:8000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip
        time.sleep(5)
        send_requests(port)
        check_workers()
        log("PASS")
    except (KeyboardInterrupt, SystemExit):
        # Cancelled or timed out: the runner kills the process seconds later, so clean up quickly.
        interrupted = True
        raise
    finally:
        if port_forward:
            port_forward.terminate()
        try:
            if not interrupted:
                save_logs()
        finally:
            wait = ["--wait=false"] if interrupted else ["--wait=true", "--timeout=10m"]
            NAMESPACE_DELETED.set()  # the heartbeat keeps renewing the lock, but stops annotating the namespace
            kubectl("delete", "namespace", NAMESPACE, "--ignore-not-found", *wait, check=False)
            if HOLDING_LOCK:
                try:
                    scale_down_if_idle()
                finally:
                    release_lock()
            stop.set()


if __name__ == "__main__":
    configure()
    # The runner stops a cancelled job with SIGINT, then SIGTERM: both run the cleanup in main().
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    main()
