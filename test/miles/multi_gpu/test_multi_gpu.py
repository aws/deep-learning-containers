"""Single-node multi-GPU tests for the Miles image.

Launches one GPU instance from a capacity reservation (the first instance type in
MULTI_GPU_INSTANCE_TYPES with free capacity wins), then runs each payload in scripts/
inside the image under test, across every GPU the instance has, followed by upstream's
short GRPO runs from the Miles checkout in the image and, for the EC2 image
(MILES_CUSTOMER_TYPE=ec2), its checkpoint-resume and SFT runs.

Usage:
    TEST_IMAGE_URI=<ecr-image-uri> MILES_TEST_MODEL_DIR=<hf-model-dir> [HF_TOKEN=<token>] \
        pytest test/miles/multi_gpu/test_multi_gpu.py -v
"""

import io
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
# Where the GRPO runs import upstream's test harness from, as in upstream CI.
UPSTREAM_PYTHONPATH = "/opt/miles:/opt/sglang/python:/opt/Megatron-LM"
LOG_TAIL_LINES = 400


@pytest.fixture(scope="module")
def gpu_host():
    """One reserved GPU instance with the payloads staged and a container running."""
    with gpu_instance() as (conn, instance_type, num_gpus):
        # SFTP does not expand ~; use paths relative to the SSH home.
        conn.run("mkdir -p ~/miles/scripts ~/miles/model")
        for script in sorted(os.listdir(SCRIPTS_DIR)):
            conn.put(os.path.join(SCRIPTS_DIR, script), f"miles/scripts/{script}")
        for name in sorted(os.listdir(MODEL_DIR)):
            if os.path.isfile(os.path.join(MODEL_DIR, name)):
                conn.put(os.path.join(MODEL_DIR, name), f"miles/model/{name}")
        # The GRPO runs download from Hugging Face. The token goes in a file outside the
        # mounted directory so that it never appears on a logged command line.
        hf_token = os.environ.get("HF_TOKEN", "")
        conn.put(
            io.BytesIO(f"HF_TOKEN={hf_token}\n".encode() if hf_token else b""), "container.env"
        )
        conn.run("chmod 600 ~/container.env")

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
            f"--env-file $HOME/container.env "
            f"-e MILES_TEST_MODEL_DIR=/miles/model {IMAGE_URI} sleep infinity"
        )
        conn.run("rm -f ~/container.env")

        gpu_name = conn.run(
            "nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1", hide=True
        ).stdout.strip()
        summary = f"Multi-GPU tests ran on {instance_type}: {num_gpus} x {gpu_name}"
        LOGGER.info(summary)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write(f"{summary}\n")
        try:
            yield conn, instance_type, num_gpus
        finally:
            conn.run(f"docker rm -f {CONTAINER_NAME}", warn=True, hide=True)


def run_in_container(conn, command, exec_args="", timeout=1800):
    result = conn.run(
        f"docker exec {exec_args} {CONTAINER_NAME} miles-entrypoint {command}",
        timeout=timeout,
        warn=True,
    )
    for name, output in (("stdout", result.stdout), ("stderr", result.stderr)):
        tail = "\n".join(output.splitlines()[-LOG_TAIL_LINES:])
        LOGGER.info(f"{command} {name} (last {LOG_TAIL_LINES} lines):\n{tail}")
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


@pytest.mark.parametrize(
    "script",
    [
        # Megatron trains on two GPUs while SGLang serves rollouts on two others, and the
        # trainer pushes new weights to the engines after every step.
        "test_qwen2.5_0.5B_gsm8k_async_short.py",
        # Training and rollout share two GPUs.
        "test_qwen2.5_0.5B_gsm8k_short.py",
    ],
    ids=["disaggregated", "colocated"],
)
def test_upstream_grpo_run(gpu_host, script):
    """Upstream's short Qwen2.5-0.5B GRPO run passes. Its --ci-test mode checks that the
    trainer's log-probs match SGLang's and that every engine picks up the new weights."""
    run_upstream_test(gpu_host, f"tests/e2e/short/{script}")


# The SageMaker image has the same training stack, so only the EC2 image spends GPU time
# on these.
@pytest.mark.skipif(os.environ.get("MILES_CUSTOMER_TYPE") != "ec2", reason="EC2 image only")
@pytest.mark.parametrize(
    "script",
    [
        # Saves a checkpoint every step, resumes from it and checks that the model hashes
        # match, then repeats with asynchronous saving.
        "tests/e2e/ckpt/test_qwen3_0.6B_ckpt.py",
        # Pure SFT, evaluated on GSM8K from the snapshots the trainer exports.
        "tests/e2e/short/test_qwen3_0.6B_sft_snapshot_eval.py",
    ],
    ids=["checkpoint-resume", "sft"],
)
def test_upstream_workflow_run(gpu_host, script):
    """Upstream's short Qwen3-0.6B run of a training workflow beyond the GRPO loop passes
    its --ci-test checks."""
    run_upstream_test(gpu_host, script)


def run_upstream_test(gpu_host, script):
    conn, instance_type, _ = gpu_host
    result = run_in_container(
        conn,
        f"python {script}",
        exec_args=f"-w /opt/miles -e PYTHONPATH={UPSTREAM_PYTHONPATH}",
        timeout=3600,
    )
    assert result.ok, f"{script} failed on {instance_type} (exit {result.return_code})"
