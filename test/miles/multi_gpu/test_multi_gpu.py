"""Single-node multi-GPU tests for the Miles image.

Launches one GPU instance from a capacity reservation (the first instance type in
MULTI_GPU_INSTANCE_TYPES with free capacity wins), then runs each payload in scripts/
inside the image under test, across every GPU the instance has.

Usage:
    TEST_IMAGE_URI=<ecr-image-uri> MILES_TEST_MODEL_DIR=<hf-model-dir> \
        pytest test/miles/multi_gpu/test_multi_gpu.py -v
"""

import logging
import os

import pytest
from test_utils.gpu_instance import gpu_instance

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

IMAGE_URI = os.environ["TEST_IMAGE_URI"]
# A small Hugging Face model on this host, copied to the instance for the SGLang test.
MODEL_DIR = os.environ["MILES_TEST_MODEL_DIR"]
CONTAINER_NAME = "miles_multi_gpu_test"
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
SUCCESS_MARKER = "MILES_MULTI_GPU_OK"


@pytest.fixture(scope="module")
def gpu_host():
    """One reserved GPU instance with the payloads and model staged and a container running."""
    with gpu_instance() as (conn, instance_type, num_gpus):
        # SFTP does not expand ~; use paths relative to the SSH home.
        conn.run("mkdir -p ~/miles/scripts ~/miles/model")
        for script in sorted(os.listdir(SCRIPTS_DIR)):
            conn.put(os.path.join(SCRIPTS_DIR, script), f"miles/scripts/{script}")
        for name in sorted(os.listdir(MODEL_DIR)):
            if os.path.isfile(os.path.join(MODEL_DIR, name)):
                conn.put(os.path.join(MODEL_DIR, name), f"miles/model/{name}")

        region = IMAGE_URI.split(".")[3]
        registry = IMAGE_URI.split("/")[0]
        conn.run(
            f"aws ecr get-login-password --region {region} "
            f"| docker login --username AWS --password-stdin {registry}",
            hide=True,
        )
        conn.run(f"docker pull {IMAGE_URI}", timeout=1800)

        # The image entrypoint sets up CUDA forward compatibility. No ports are published.
        conn.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)
        conn.run(
            f"docker run -d --rm --init --gpus all --ipc=host --ulimit memlock=-1 "
            f"--name {CONTAINER_NAME} -v $HOME/miles:/miles:ro "
            f"-e MILES_TEST_MODEL_DIR=/miles/model {IMAGE_URI} sleep infinity"
        )
        LOGGER.info(f"Container ready on {instance_type} with {num_gpus} GPUs")
        try:
            yield conn, instance_type, num_gpus
        finally:
            conn.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)


def run_in_container(conn, command):
    result = conn.run(
        f"docker exec {CONTAINER_NAME} miles-entrypoint {command}", timeout=1800, warn=True
    )
    LOGGER.info(f"{command} stdout:\n{result.stdout}")
    LOGGER.info(f"{command} stderr:\n{result.stderr}")
    return result


def test_nccl_tests_all_reduce_is_correct(gpu_host):
    """The shipped nccl-tests binary all-reduces across every GPU with correct results."""
    conn, instance_type, num_gpus = gpu_host
    result = run_in_container(conn, f"all_reduce_perf -b 8 -e 256M -f 2 -g {num_gpus} -c 1")
    assert result.ok, f"all_reduce_perf failed on {instance_type} (exit {result.return_code})"
    assert "Out of bounds values : 0 OK" in result.stdout


@pytest.mark.parametrize("payload, port", [("collectives", 29500), ("megatron_parallel", 29501)])
def test_torchrun_payload(gpu_host, payload, port):
    """Run one torchrun payload with a rank on every GPU."""
    conn, instance_type, num_gpus = gpu_host
    result = run_in_container(
        conn,
        f"torchrun --nproc_per_node={num_gpus} --master_port={port} /miles/scripts/{payload}.py",
    )
    assert result.ok, f"{payload} failed on {instance_type} with {num_gpus} GPUs"
    assert SUCCESS_MARKER in result.stdout, f"{payload} did not report success"


def test_sglang_tensor_parallel_serving(gpu_host):
    """SGLang serves the model split across two GPUs and answers correctly."""
    conn, instance_type, _ = gpu_host
    result = run_in_container(conn, "python /miles/scripts/sglang_tensor_parallel.py --tp 2")
    assert result.ok, f"SGLang tensor-parallel serving failed on {instance_type}"
    assert SUCCESS_MARKER in result.stdout
