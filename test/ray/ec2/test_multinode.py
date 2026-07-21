# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Multi-node replica-distribution test for Ray DLC.

Deploys cv-densenet across two containers (head + worker on a shared Docker
network) with `num_replicas: 2, max_replicas_per_node: 1` — one full replica of
the model per node (data-parallel; each node serves requests independently, no
model sharding). Verifies:
  1. A replica is placed on each of the two distinct Ray nodes (i.e. the worker
     joined the cluster and Serve scheduled a replica onto it).
  2. Inference requests against the real cv-densenet model return valid
     predictions through the multi-node deployment.

The head container uses the DLC's default entrypoint (single-node Ray Serve
runtime). The worker container uses the entrypoint's new RAY_ROLE=worker branch
to join the head instead of starting its own head.
"""

import json
import logging
import os
import shutil
import subprocess
import time
import uuid

import pytest
import yaml
from ray.ec2.common import (
    DEFAULT_SERVE_PORT,
    HEALTH_INTERVAL,
    HEALTH_TIMEOUT,
    download_and_extract_model,
    post_bytes,
    wait_for_health,
)
from ray.utils import download_all_test_images, validate_densenet_response

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

REPLICA_READY_TIMEOUT = 300


def _run(cmd, check=True):
    LOGGER.info("+ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _docker_logs(container_id):
    result = subprocess.run(["docker", "logs", container_id], capture_output=True, text=True)
    return result.stdout + result.stderr


def _write_multinode_config(model_dir, num_replicas=2):
    """Overwrite the model's config.yaml with a multi-node Ray Serve config.

    Pins autoscaling to exactly `num_replicas` (min == max) and sets
    `max_replicas_per_node: 1` to force replicas onto distinct nodes, with
    `proxy_location: EveryNode` so HTTP proxies run on both nodes. Preserves
    the deployment's import_path from the original config.
    """
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path) as f:
        original = yaml.safe_load(f)

    # Original schema is Ray Serve v2 (applications[]).
    app = original["applications"][0]
    for deployment in app.get("deployments", []):
        # A config-level `num_replicas` does NOT override a decorator-level
        # `autoscaling_config` (as cv-densenet's deployment.py sets), and the two
        # are mutually exclusive. So pin the autoscaling floor+ceiling instead:
        # min == max == num_replicas forces exactly that many replicas, and this
        # config-level autoscaling_config overrides whatever the decorator set.
        deployment.pop("num_replicas", None)
        deployment["autoscaling_config"] = {
            "min_replicas": num_replicas,
            "max_replicas": num_replicas,
        }
        deployment["max_replicas_per_node"] = 1

    original["proxy_location"] = "EveryNode"
    original.setdefault("http_options", {})
    original["http_options"]["host"] = "0.0.0.0"
    original["http_options"]["port"] = DEFAULT_SERVE_PORT

    with open(config_path, "w") as f:
        yaml.safe_dump(original, f)

    LOGGER.info("Wrote multi-node config to %s", config_path)


@pytest.fixture(scope="function")
def multinode_cluster(aws_session, image_uri):
    """Start head + worker on a shared Docker network with cv-densenet mounted.

    Head runs the default DLC entrypoint (Ray Serve auto-deploys the mounted
    config). Worker joins via RAY_ROLE=worker + RAY_HEAD_ADDRESS.

    Note: the head auto-deploys as soon as it starts, before the worker has
    joined. With max_replicas_per_node=1 the second replica stays PENDING until
    the worker joins, so REPLICA_READY_TIMEOUT must comfortably exceed worker
    startup + Ray join time. wait_for_health only confirms the proxy is up (one
    replica), not full replica distribution — that is what _wait_for_distributed_replicas checks.
    """
    suffix = uuid.uuid4().hex[:8]
    network = f"ray-mn-{suffix}"
    head_name = f"ray-head-{suffix}"
    worker_name = f"ray-worker-{suffix}"
    model_dir = download_and_extract_model(aws_session, "cv-densenet", "cpu")
    _write_multinode_config(model_dir, num_replicas=2)

    _run(["docker", "network", "create", network])

    try:
        # Head: default entrypoint, auto-detects /opt/ml/model/config.yaml.
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                head_name,
                "--network",
                network,
                "--shm-size=2g",
                "-p",
                f"{DEFAULT_SERVE_PORT}:{DEFAULT_SERVE_PORT}",
                "-v",
                f"{model_dir}:/opt/ml/model",
                "-e",
                "RAY_SERVE_HTTP_HOST=0.0.0.0",
                image_uri,
            ]
        )

        # Worker: joins head via new RAY_ROLE=worker branch. Mount the same model
        # dir so a replica scheduled here can import the deployment module
        # (the worker entrypoint puts /opt/ml/model on PYTHONPATH).
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                worker_name,
                "--network",
                network,
                "--shm-size=2g",
                "-v",
                f"{model_dir}:/opt/ml/model",
                "-e",
                "RAY_ROLE=worker",
                "-e",
                f"RAY_HEAD_ADDRESS={head_name}:6379",
                image_uri,
            ]
        )

        # Wait for HTTP endpoint to become healthy (Serve is up).
        try:
            wait_for_health(
                port=DEFAULT_SERVE_PORT,
                timeout=HEALTH_TIMEOUT,
                interval=HEALTH_INTERVAL,
            )
        except TimeoutError:
            LOGGER.error("Head logs:\n%s", _docker_logs(head_name))
            LOGGER.error("Worker logs:\n%s", _docker_logs(worker_name))
            raise

        # Wait for both replicas to be placed on distinct nodes.
        _wait_for_distributed_replicas(head_name)

        yield {
            "head": head_name,
            "worker": worker_name,
            "network": network,
            "model_dir": model_dir,
        }

    finally:
        for name in (worker_name, head_name):
            LOGGER.info("--- %s logs ---\n%s", name, _docker_logs(name))
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "network", "rm", network], capture_output=True)
        shutil.rmtree(model_dir, ignore_errors=True)


# Per-replica node placement is NOT on serve.status() (its deployments expose
# only a replica_states count map). Use the detail API ServeInstanceDetails,
# whose ReplicaDetails carry replica_id / state / node_id.
_REPLICA_QUERY = (
    "import json\n"
    "from ray import serve\n"
    "details = serve.context._get_global_client().get_serve_details()\n"
    "inst = serve.schema.ServeInstanceDetails(**details)\n"
    "out = []\n"
    "for app in inst.applications.values():\n"
    "    for dep in app.deployments.values():\n"
    "        for r in dep.replicas:\n"
    "            out.append((r.replica_id, r.state, r.node_id))\n"
    "print(json.dumps(out))"
)


def _query_running_replicas(head_name):
    """Return list of (replica_id, node_id) for RUNNING replicas via the head.

    Returns None if the query command failed (logs stderr for diagnostics).
    """
    result = subprocess.run(
        ["docker", "exec", head_name, "python", "-c", _REPLICA_QUERY],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        LOGGER.warning("replica query failed (rc=%s): %s", result.returncode, result.stderr.strip())
        return None
    try:
        replicas = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        LOGGER.warning("replica query parse failed: %s (stdout=%r)", e, result.stdout)
        return None
    return [(rid, node) for rid, state, node in replicas if state == "RUNNING"]


def _wait_for_distributed_replicas(head_name):
    """Poll until >=2 replicas are RUNNING on distinct node IDs; return the node set."""
    deadline = time.time() + REPLICA_READY_TIMEOUT
    while time.time() < deadline:
        running = _query_running_replicas(head_name)
        if running is not None:
            node_ids = {node for _, node in running}
            LOGGER.info("Running replicas: %s (unique nodes=%d)", running, len(node_ids))
            if len(running) >= 2 and len(node_ids) >= 2:
                return node_ids
        time.sleep(5)
    raise TimeoutError(f"2 replicas on distinct nodes not ready within {REPLICA_READY_TIMEOUT}s")


def test_multinode_replica_distribution(multinode_cluster):
    """cv-densenet replicas spread one-per-node across head + worker, and the
    real model serves valid predictions through the multi-node deployment."""
    head = multinode_cluster["head"]

    # 1. Verify replica distribution: one replica placed on each of the two
    #    distinct nodes (confirms the worker joined and Serve scheduled a
    #    replica onto it, not just onto the head).
    running = _query_running_replicas(head)
    assert running is not None, "replica placement query failed (see logged stderr)"
    node_ids = {node for _, node in running}
    assert len(node_ids) == 2, f"expected replicas on 2 nodes, got {len(node_ids)} ({running})"

    # 2. Send real cv-densenet inference requests and validate predictions.
    #    Repeat each image several times so requests fan out across both nodes'
    #    replicas (not just a single call that one replica could fully serve).
    images = download_all_test_images()
    for img_name, img_data in images.items():
        for _ in range(5):
            response = post_bytes(img_data, "image/jpeg")
            err = validate_densenet_response(response)
            assert not err, f"cv-densenet {img_name}: {err}"

        top = response["predictions"][0]
        LOGGER.info(
            "cv-densenet %s -> %s (class_id=%s, prob=%.4f)",
            img_name,
            top["class_name"],
            top["class_id"],
            top["probability"],
        )
