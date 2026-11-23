"""Hugging Face inference on Trainium2 with the native Torch NeuronX backend.

Each case runs in eager mode and with `torch.compile(backend="neuron")`:
- transformers_neuron_inference.py checks Transformers (encoder, decoder and
  generation), Sentence Transformers, a Diffusers UNet and a tiny Stable
  Diffusion pipeline (end to end) against CPU.
- hf_serve_neuron_inference.py serves models through the image entrypoint
  (hf-serve) with zero-code tasks (text-classification, text-to-image) and a
  custom handler.
"""

import pytest

from packaging.version import Version

from test import test_utils
from test.test_utils import ec2


NEURON_INSTANCE_TYPES = ec2.get_ec2_instance_type(
    default="trn2.3xlarge", processor="neuronx", job_type="inference"
)


@pytest.fixture(autouse=True)
def native_huggingface_image(pytorch_inference_neuronx):
    _, version = test_utils.get_framework_and_version_from_tag(pytorch_inference_neuronx)
    if Version(version) < Version("2.13"):
        pytest.skip("Native Torch NeuronX inference requires a PyTorch 2.13+ image")


def _run_in_container(ec2_connection, image, region, script, mode):
    # The container supplies userspace Neuron libraries; the host needs a current driver.
    ec2_connection.run(
        "sudo apt-get update && sudo apt-get install -y aws-neuronx-dkms", timeout=600
    )
    container_name = f"hf_neuron_inference_{script.split('_')[0]}_{mode}"
    ec2.execute_ec2_training_test(
        ec2_connection,
        image,
        f"python3 /test/bin/huggingface/{script} --mode {mode}",
        region=region,
        large_shm=True,
        container_name=container_name,
        timeout=3600,
        bin_bash_entrypoint=True,
    )


@pytest.mark.usefixtures("huggingface_only")
@pytest.mark.processor("neuronx")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
@pytest.mark.model("tiny-bert-qwen3-stable-diffusion")
@pytest.mark.integration("huggingface_neuron_inference")
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL22_BASE_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", NEURON_INSTANCE_TYPES, indirect=True)
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_huggingface_neuron_inference(pytorch_inference_neuronx, ec2_connection, region, mode):
    _run_in_container(
        ec2_connection,
        pytorch_inference_neuronx,
        region,
        "transformers_neuron_inference.py",
        mode,
    )


@pytest.mark.usefixtures("huggingface_only")
@pytest.mark.processor("neuronx")
@pytest.mark.neuronx_test
@pytest.mark.team("neuron")
@pytest.mark.model("tiny-bert-stable-diffusion")
@pytest.mark.integration("huggingface_neuron_hf_serve")
@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL22_BASE_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", NEURON_INSTANCE_TYPES, indirect=True)
@pytest.mark.parametrize("mode", ["eager", "compiled"])
def test_huggingface_neuron_hf_serve(pytorch_inference_neuronx, ec2_connection, region, mode):
    _run_in_container(
        ec2_connection,
        pytorch_inference_neuronx,
        region,
        "hf_serve_neuron_inference.py",
        mode,
    )
