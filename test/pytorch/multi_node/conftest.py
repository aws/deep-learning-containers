"""Fixtures for multi-node tests using a Docker bridge network with 2 containers."""

import subprocess
import uuid

import pytest

NETWORK_PREFIX = "dlc-test"


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=kwargs.get("timeout", 60))


@pytest.fixture(scope="module")
def multinode_cluster(image_uri):
    """Spin up 2 GPU containers on a shared Docker network.

    Yields a helper with .exec(node, cmd) and .get_logs(node) methods.
    Cleans up containers and network on teardown.
    """
    net = f"{NETWORK_PREFIX}-{uuid.uuid4().hex[:8]}"
    nodes = ["node0", "node1"]
    containers = {}

    # Create network
    _run(["docker", "network", "create", net])

    try:
        # Start 2 detached containers with GPUs, sleeping forever
        for name in nodes:
            result = _run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    f"--name={name}-{net}",
                    f"--hostname={name}",
                    f"--network={net}",
                    "--gpus=all",
                    "--shm-size=1g",
                    image_uri,
                    "sleep",
                    "infinity",
                ]
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start {name}: {result.stderr}")
            containers[name] = f"{name}-{net}"

        class Cluster:
            def exec(self, node, cmd, timeout=120):
                """Run a command in the specified node container."""
                cid = containers[node]
                result = subprocess.run(
                    ["docker", "exec", cid, "bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return result

            def get_logs(self, node):
                cid = containers[node]
                result = _run(["docker", "logs", cid])
                return result.stdout

        yield Cluster()

    finally:
        # Cleanup: stop containers, remove network
        for cid in containers.values():
            _run(["docker", "kill", cid])
        _run(["docker", "network", "rm", net])
