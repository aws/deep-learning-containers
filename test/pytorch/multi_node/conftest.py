"""Fixtures for multi-node tests.

The workflow creates a Docker bridge network with 2 GPU containers and sets:
  NODE0_CONTAINER_ID, NODE1_CONTAINER_ID
"""

import os
import subprocess

import pytest


@pytest.fixture(scope="module")
def multinode_cluster():
    """Provide exec access to the 2 containers started by the workflow."""
    containers = {
        "node0": os.environ.get("NODE0_CONTAINER_ID"),
        "node1": os.environ.get("NODE1_CONTAINER_ID"),
    }
    if not all(containers.values()):
        pytest.skip("NODE0_CONTAINER_ID / NODE1_CONTAINER_ID not set — run via workflow")

    class Cluster:
        def exec(self, node, cmd, timeout=120):
            cid = containers[node]
            return subprocess.run(
                ["docker", "exec", cid, "bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

    yield Cluster()
