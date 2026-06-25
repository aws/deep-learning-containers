"""EC2 EFA integration test.

Launches 2x p4d.24xlarge with EFA, runs NCCL all_reduce_perf across nodes,
and verifies EFA transport is used (not sockets).

When RUN_NIXL_TESTS=1, additionally exercises NIXL's libfabric backend
(packaging smoke test + multi-node disaggregated prefill/decode).

Ported from V1: test/dlc_tests/ec2/test_efa.py + test/v2/ec2/efa/test_efa.py

Usage:
    pytest test/efa/test_efa.py --image-uri <ecr-image-uri> -v
"""

import os

from test_utils.efa_helpers import (
    DEFAULT_TIMEOUT,
    HOSTS_FILE_LOCATION,
    MASTER_CONTAINER_NAME,
    WORKER_CONTAINER_NAME,
    efa_instances,
    run_on_container,
)

IMAGE_URI = os.environ["TEST_IMAGE_URI"]
EFA_INSTANCE_TYPE = os.environ.get("EFA_INSTANCE_TYPE", "p4d.24xlarge")
RUN_NIXL_TESTS = os.environ.get("RUN_NIXL_TESTS", "0") == "1"
# Disagg-PD orchestration is vLLM-specific; set 0 to run the libfabric smoke only.
RUN_NIXL_DISAGG = os.environ.get("RUN_NIXL_DISAGG", "1") == "1"
NIXL_MODEL = os.environ.get("NIXL_TEST_MODEL", "facebook/opt-125m")

# Container launch options. Defaults work for PyTorch DLCs (use the image's
# entrypoint, pass `bash` as argv so the entrypoint's `exec "$@"` runs bash
# under -id stdin). Callers testing images whose baked-in entrypoint can't
# accept a generic shell arg (e.g., vLLM's `vllm serve "$@"`) override via
# env: EFA_CONTAINER_ENTRYPOINT=/bin/bash EFA_CONTAINER_CMD="-c 'sleep infinity'".
EFA_CONTAINER_ENTRYPOINT = os.environ.get("EFA_CONTAINER_ENTRYPOINT") or None
EFA_CONTAINER_CMD = os.environ.get("EFA_CONTAINER_CMD", "bash")

# vLLM startup + KV transfer + accuracy check.
NIXL_DISAGG_TIMEOUT = 1200


def test_efa_sanity_and_nccl(image_uri=IMAGE_URI):
    """Run EFA sanity checks and NCCL all_reduce_perf across 2 nodes.

    Verifies:
    - EFA provider detected (fi_info -p efa)
    - fi_pingpong over EFA loopback
    - RDMA devices present (ibv_devinfo)
    - GPU Direct RDMA (GDR) available
    - NCCL uses EFA transport ("Selected provider is efa")
    - NCCL uses Libfabric ("Using network Libfabric")
    - NCCL uses GDRDMA on p4d/p5 ("NET/Libfabric/0/GDRDMA")
    - all_reduce bandwidth >= 3 GB/s on the 1 GiB message size
    """

    # On success, run_on_container returns silently and the script's stdout
    # never reaches pytest's captured log — only the LOGGER.info "Running on"
    # line shows up. Wrap every step so the cmd, full stdout/stderr, and exit
    # code are visible in the test log regardless of pass/fail. pytest is run
    # with -s so prints land in the captured-stdout section.
    def _step(name, container, conn, cmd, timeout=DEFAULT_TIMEOUT):
        print(f"\n========== {name} ==========")
        print(f"$ {cmd}")
        r = run_on_container(container, conn, cmd, timeout=timeout)
        if r.stdout:
            print(f"--- stdout ({len(r.stdout)} chars) ---")
            print(r.stdout)
        if r.stderr:
            print(f"--- stderr ({len(r.stderr)} chars) ---")
            print(r.stderr)
        print(f"========== /{name} (exit={r.exited}) ==========\n")
        return r

    with efa_instances(
        image_uri=image_uri,
        instance_type=EFA_INSTANCE_TYPE,
        container_entrypoint=EFA_CONTAINER_ENTRYPOINT,
        container_cmd=EFA_CONTAINER_CMD,
    ) as (
        master_conn,
        worker_conn,
        aws_session,
    ):
        # No-op for PyTorch DLCs (binary is preinstalled); apt-installs
        # libnccl-dev (if missing) and compiles nccl-tests for vLLM Ubuntu.
        # verifiable.cu is template-heavy and the build legitimately takes
        # ~13 min, hence the larger timeout vs DEFAULT_TIMEOUT (600s).
        for name, conn in (
            (MASTER_CONTAINER_NAME, master_conn),
            (WORKER_CONTAINER_NAME, worker_conn),
        ):
            _step(
                f"setup_nccl_tests:{name}",
                name,
                conn,
                "/test/efa/scripts/setup_nccl_tests.sh",
                timeout=1500,
            )

        _step(
            "efa_sanity",
            MASTER_CONTAINER_NAME,
            master_conn,
            "/test/efa/scripts/efa_sanity.sh",
        )

        _step(
            "nccl_allreduce",
            MASTER_CONTAINER_NAME,
            master_conn,
            f"/test/efa/scripts/nccl_allreduce.sh {HOSTS_FILE_LOCATION} 2",
            timeout=DEFAULT_TIMEOUT,
        )

        if not RUN_NIXL_TESTS:
            return

        # Smoke: LIBFABRIC plugin loads and binds to the EFA libfabric provider.
        # Cheap regression catch for nixl-cu* wheel packaging issues.
        _step(
            "nixl:libfabric_smoke",
            MASTER_CONTAINER_NAME,
            master_conn,
            "python3 /test/efa/scripts/nixl_libfabric_smoke.py",
        )

        if not RUN_NIXL_DISAGG:
            return

        # Disaggregated prefill/decode across both nodes with NIXL+LIBFABRIC.
        # The worker's private IP is already in the MPI hosts file written by
        # the fixture (line 1 = localhost, line 2 = "<worker_ip> slots=N").
        # run_on_container wraps the cmd in bash -c '<cmd>', so any single
        # quotes inside cmd break the wrapping. Read the raw hosts file
        # contents back and parse here, avoiding shell quoting entirely.
        # File format: "localhost slots=N\n<worker_ip> slots=N".
        hosts_contents = run_on_container(
            MASTER_CONTAINER_NAME,
            master_conn,
            f"cat {HOSTS_FILE_LOCATION}",
        ).stdout
        worker_ip = hosts_contents.splitlines()[1].split()[0]
        print(f"NIXL: parsed worker_ip={worker_ip} from hosts file")

        _step(
            "nixl:decode_launch",
            WORKER_CONTAINER_NAME,
            worker_conn,
            f"/test/efa/scripts/nixl_disagg_pd_decode.sh {NIXL_MODEL}",
        )
        # Always dump prefill/decode/proxy logs from inside the containers —
        # they only live on the master/worker container filesystems and are
        # gone once the fixture terminates the EC2 instances. Wrap in
        # try/finally so the dump fires even when the orchestrator fails
        # (which is the case where we need them most).
        try:
            _step(
                "nixl:disagg_pd_orchestrator",
                MASTER_CONTAINER_NAME,
                master_conn,
                f"/test/efa/scripts/nixl_disagg_pd.sh {worker_ip} {NIXL_MODEL}",
                timeout=NIXL_DISAGG_TIMEOUT,
            )
        finally:
            for name, container, conn, log_path in (
                ("prefill", MASTER_CONTAINER_NAME, master_conn, "/test/efa/logs/prefill.log"),
                ("proxy", MASTER_CONTAINER_NAME, master_conn, "/test/efa/logs/proxy.log"),
                ("decode", WORKER_CONTAINER_NAME, worker_conn, "/test/efa/logs/decode.log"),
            ):
                print(f"\n========== {name} log ({log_path}) ==========")
                try:
                    r = run_on_container(container, conn, f"cat {log_path}", warn=True)
                    print(r.stdout if r.stdout else "(empty)")
                except Exception as e:  # noqa: BLE001
                    print(f"(could not read: {e})")
                print(f"========== /{name} log ==========\n")
