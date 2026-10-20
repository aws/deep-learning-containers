"""Two-node tests for the Miles image.

Launches two EFA instances from a capacity reservation with the image running on each, as
the shared EFA suite does (see test_utils/efa_helpers.py), joins the two containers into
one Ray cluster, and runs scripts/grpo_two_node.py on it. Megatron trains on one node
while SGLang serves rollouts on the other, so every weight update crosses EFA.

Usage:
    TEST_IMAGE_URI=<ecr-image-uri> [HF_TOKEN=<token>] \
        pytest test/miles/multi_node/test_multi_node.py -v
"""

import io
import logging
import os
import shlex

import pytest
from test_utils.efa_helpers import MASTER_CONTAINER_NAME, WORKER_CONTAINER_NAME, efa_instances

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

IMAGE_URI = os.environ["TEST_IMAGE_URI"]
INSTANCE_TYPE = os.environ.get("MULTI_NODE_INSTANCE_TYPE", "p4d.24xlarge")
GPUS_PER_NODE = 8
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "grpo_two_node.py")
# Where the GRPO run imports upstream's test harness from, as in upstream CI.
UPSTREAM_PYTHONPATH = "/opt/miles:/opt/sglang/python:/opt/Megatron-LM"
# efa_instances mounts each host's ~/test at /test in its container.
STAGE_DIR = "test/miles_multi_node"
# The token goes in a file so that it never appears on a logged command line.
EXPORT_HF_TOKEN = (
    f'if [ -s /{STAGE_DIR}/hf_token ]; then export HF_TOKEN="$(cat /{STAGE_DIR}/hf_token)"; fi; '
)
NCCL_LOG_DIR = "/tmp/nccl-logs"
LOG_TAIL_LINES = 400


def private_ip(conn):
    """The instance's primary private IP, from instance metadata."""
    return conn.run(
        "TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token "
        '-H "X-aws-ec2-metadata-token-ttl-seconds: 60") && '
        'curl -s -H "X-aws-ec2-metadata-token: $TOKEN" '
        "http://169.254.169.254/latest/meta-data/local-ipv4",
        hide=True,
    ).stdout.strip()


def run_in_container(conn, container, command, env=None, timeout=600):
    """Run a shell command in the container through the image entrypoint."""
    env_args = " ".join(f"-e {name}={shlex.quote(value)}" for name, value in (env or {}).items())
    result = conn.run(
        f"docker exec -w /opt/miles -e PYTHONPATH={UPSTREAM_PYTHONPATH} {env_args} {container} "
        f"miles-entrypoint bash -c {shlex.quote(command)}",
        timeout=timeout,
        warn=True,
    )
    for name, output in (("stdout", result.stdout), ("stderr", result.stderr)):
        tail = "\n".join(output.splitlines()[-LOG_TAIL_LINES:])
        LOGGER.info(f"{container}: {command} {name} (last {LOG_TAIL_LINES} lines):\n{tail}")
    return result


@pytest.fixture(scope="module")
def ray_cluster():
    """Two EFA instances whose containers form one Ray cluster."""
    with efa_instances(IMAGE_URI, instance_type=INSTANCE_TYPE) as (head_conn, worker_conn, _):
        head_ip, worker_ip = private_ip(head_conn), private_ip(worker_conn)
        hf_token = os.environ.get("HF_TOKEN", "")
        for conn in (head_conn, worker_conn):
            # SFTP does not expand ~; use paths relative to the SSH home.
            conn.run(f"mkdir -p ~/{STAGE_DIR}")
            conn.put(SCRIPT, f"{STAGE_DIR}/grpo_two_node.py")
            conn.put(io.BytesIO(hf_token.encode()), f"{STAGE_DIR}/hf_token")
            conn.run(f"chmod 600 ~/{STAGE_DIR}/hf_token")

        num_gpus, gpu_name = (
            head_conn.run("nvidia-smi --query-gpu=name --format=csv,noheader | uniq -c", hide=True)
            .stdout.strip()
            .split(maxsplit=1)
        )
        assert int(num_gpus) == GPUS_PER_NODE, f"{INSTANCE_TYPE} has {num_gpus} GPUs"
        summary = f"Multi-node tests ran on 2 x {INSTANCE_TYPE}: {num_gpus} x {gpu_name} each"
        LOGGER.info(summary)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(f"{summary}\n")

        # Host networking: Ray listens on the instances' own interfaces, which their
        # security groups open only to each other and to SSH from this runner.
        head = run_in_container(
            head_conn,
            MASTER_CONTAINER_NAME,
            f"mkdir -p {NCCL_LOG_DIR} && ray start --head --node-ip-address {head_ip} "
            f"--port 6379 --num-gpus {GPUS_PER_NODE} --disable-usage-stats",
        )
        assert head.ok, "The Ray head did not start"
        worker = run_in_container(
            worker_conn,
            WORKER_CONTAINER_NAME,
            f"mkdir -p {NCCL_LOG_DIR} && ray start --address {head_ip}:6379 "
            f"--node-ip-address {worker_ip} --num-gpus {GPUS_PER_NODE} --disable-usage-stats",
        )
        assert worker.ok, "The Ray worker did not join the head"
        try:
            yield head_conn, worker_conn, head_ip
        finally:
            run_in_container(worker_conn, WORKER_CONTAINER_NAME, "ray stop --force")
            run_in_container(head_conn, MASTER_CONTAINER_NAME, "ray stop --force")


def test_grpo_across_nodes_broadcasts_weights_over_efa(ray_cluster):
    """Upstream's short GRPO run passes with training and rollout on different nodes, and
    the trainer's weight broadcast reaches the other node over EFA."""
    head_conn, worker_conn, head_ip = ray_cluster
    prepared = run_in_container(
        worker_conn,
        WORKER_CONTAINER_NAME,
        f"{EXPORT_HF_TOKEN}python /{STAGE_DIR}/grpo_two_node.py prepare",
        timeout=1800,
    )
    assert prepared.ok, "Downloading the model and data on the worker node failed"

    result = run_in_container(
        head_conn,
        MASTER_CONTAINER_NAME,
        f"{EXPORT_HF_TOKEN}python /{STAGE_DIR}/grpo_two_node.py",
        env={
            "MILES_SCRIPT_EXTERNAL_RAY": "1",
            "MASTER_ADDR": head_ip,
            # upstream's harness forwards both to every Ray worker; %h and %p name a
            # file per host and process.
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_FILE": f"{NCCL_LOG_DIR}/%h-%p.log",
        },
        timeout=3600,
    )
    assert result.ok, f"The two-node GRPO run failed (exit {result.return_code})"

    # NCCL logs the transport of every connection it sets up. Only connections to the
    # other node go through the network: over EFA they show as Libfabric, and a fallback
    # to TCP would show as Socket.
    for conn, container in (
        (head_conn, MASTER_CONTAINER_NAME),
        (worker_conn, WORKER_CONTAINER_NAME),
    ):
        transports = run_in_container(
            conn,
            container,
            f"cat {NCCL_LOG_DIR}/*.log | grep -o 'via NET/[A-Za-z]*' | sort | uniq -c",
        ).stdout
        assert "via NET/Libfabric" in transports, (
            f"{container} made no EFA connection: {transports}"
        )
        assert "via NET/Socket" not in transports, f"{container} fell back to TCP: {transports}"
