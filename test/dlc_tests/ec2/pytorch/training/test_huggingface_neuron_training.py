"""Transformers tensor-parallel and FSDP training on a Trainium2 instance."""

import pytest

from test import test_utils
from test.test_utils import ec2


NEURON_INSTANCE_TYPES = ec2.get_ec2_instance_type(
    default="trn2.3xlarge", processor="neuronx", job_type="training"
)


@pytest.fixture(autouse=True)
def native_huggingface_image(pytorch_training_neuronx):
    _, version = test_utils.get_framework_and_version_from_tag(pytorch_training_neuronx)
    if version != "2.14.0":
        pytest.skip("Native Torch Neuron training requires the PyTorch 2.14 image")


@pytest.mark.usefixtures("huggingface_only")
@pytest.mark.processor("neuronx")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
@pytest.mark.model("qwen3")
@pytest.mark.integration("huggingface_neuron_training")
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL22_BASE_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", NEURON_INSTANCE_TYPES, indirect=True)
@pytest.mark.parametrize("parallel_mode", ["tp", "fsdp"])
def test_transformers_neuron_training(
    pytorch_training_neuronx, ec2_connection, region, parallel_mode
):
    # The container supplies userspace Neuron libraries; the host needs a current driver.
    ec2_connection.run(
        "sudo apt-get update && sudo apt-get install -y aws-neuronx-dkms", timeout=600
    )
    command = (
        "ON_NEURON_EAGER=1 HF_DEACTIVATE_ASYNC_LOAD=1 OMP_NUM_THREADS=4 "
        "TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 "
        "torchrun --standalone --nproc_per_node=2 "
        "/test/bin/huggingface/transformers_neuron_training.py "
        f"--parallel-mode {parallel_mode}"
    )
    ec2.execute_ec2_training_test(
        ec2_connection,
        pytorch_training_neuronx,
        command,
        region=region,
        large_shm=True,
        host_network=True,
        container_name=f"hf_neuron_{parallel_mode}",
        timeout=1800,
        bin_bash_entrypoint=True,
    )
