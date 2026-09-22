"""Single-node multi-GPU integration test.

Launches one GPU instance from a capacity reservation (the first candidate instance type
with free capacity wins), then runs each torchrun payload in scripts/ inside the image
under test, across every GPU the instance has.

Usage:
    pytest test/pytorch/multi_gpu/test_multi_gpu.py --image-uri <ecr-image-uri> -v
"""

import logging
import os

import pytest
from test_utils.gpu_instance import gpu_instance

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

IMAGE_URI = os.environ["TEST_IMAGE_URI"]
CONTAINER_NAME = "multi_gpu_test"
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

PAYLOADS = (("ddp", 29500), ("fsdp", 29501), ("deepspeed", 29502))


@pytest.fixture(scope="module")
def gpu_host():
    """One reserved GPU instance with the payloads staged and a container running."""
    with gpu_instance() as (conn, instance_type, num_gpus):
        # SFTP does not expand ~; use paths relative to the SSH home.
        conn.run("mkdir -p ~/test/multi_gpu/scripts")
        for script in sorted(os.listdir(SCRIPTS_DIR)):
            conn.put(os.path.join(SCRIPTS_DIR, script), f"test/multi_gpu/scripts/{script}")

        region = IMAGE_URI.split(".")[3]
        registry = IMAGE_URI.split("/")[0]
        conn.run(
            f"aws ecr get-login-password --region {region} "
            f"| docker login --username AWS --password-stdin {registry}",
            hide=True,
        )
        conn.run(f"docker pull {IMAGE_URI}", timeout=1800)

        # Image entrypoint sets CUDA forward-compat; -id keeps stdin open so bash blocks.
        conn.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)
        conn.run(
            f"docker run --gpus all -id --name {CONTAINER_NAME} --shm-size=2g "
            f"-v $HOME/test:/test -v /dev/shm:/dev/shm {IMAGE_URI} bash"
        )
        LOGGER.info(f"Container ready on {instance_type} with {num_gpus} GPUs")
        try:
            yield conn, instance_type, num_gpus
        finally:
            conn.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)


@pytest.mark.parametrize("payload,port", PAYLOADS, ids=[p[0] for p in PAYLOADS])
def test_multi_gpu_payload(gpu_host, payload, port):
    """Run one torchrun payload across every GPU on the host."""
    conn, instance_type, num_gpus = gpu_host
    result = conn.run(
        f"docker exec {CONTAINER_NAME} "
        f"torchrun --nproc_per_node={num_gpus} --master_port={port} "
        f"/test/multi_gpu/scripts/{payload}.py",
        timeout=1800,
        warn=True,
    )
    assert result.ok, (
        f"{payload} failed on {instance_type} with {num_gpus} GPUs "
        f"(exit {result.return_code})\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
    assert "ok" in result.stdout, f"{payload} did not report success:\n{result.stdout[-2000:]}"
