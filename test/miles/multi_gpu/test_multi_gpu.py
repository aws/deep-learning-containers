"""Verify the Miles image across every GPU on one node: NCCL, Megatron-Core tensor and
data parallelism, and SGLang tensor-parallel serving.

Runs inside the image under test. Each payload in scripts/ asserts on every rank and
prints a marker once every check passed. MILES_TEST_MODEL_DIR points at a Hugging Face
model directory for the SGLang test.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
SUCCESS_MARKER = "MILES_MULTI_GPU_OK"
NUM_GPUS = torch.cuda.device_count()


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    # pytest shows a test's captured output when it fails.
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result


def test_node_has_several_gpus():
    assert NUM_GPUS >= 2


def test_nccl_tests_all_reduce_is_correct():
    """The shipped nccl-tests binary all-reduces across every GPU with correct results."""
    result = run(
        ["all_reduce_perf", "-b", "8", "-e", "256M", "-f", "2", "-g", str(NUM_GPUS), "-c", "1"]
    )
    assert result.returncode == 0
    assert "Out of bounds values : 0 OK" in result.stdout


@pytest.mark.parametrize("payload, port", [("collectives", 29500), ("megatron_parallel", 29501)])
def test_torchrun_payload(payload, port):
    """Run one torchrun payload with a rank on every GPU."""
    result = run(
        [
            "torchrun",
            f"--nproc_per_node={NUM_GPUS}",
            f"--master_port={port}",
            str(SCRIPTS_DIR / f"{payload}.py"),
        ]
    )
    assert result.returncode == 0
    assert SUCCESS_MARKER in result.stdout


def test_sglang_tensor_parallel_serving():
    """SGLang serves the model split across two GPUs and answers correctly."""
    result = run([sys.executable, str(SCRIPTS_DIR / "sglang_tensor_parallel.py"), "--tp", "2"])
    assert result.returncode == 0
    assert SUCCESS_MARKER in result.stdout
