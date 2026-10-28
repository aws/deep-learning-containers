"""Run the Hugging Face Torch NeuronX image's CPU sanity suite."""

import json
from pathlib import Path
import shlex
import tempfile

import pytest
from invoke.context import Context

from src.buildspec import Buildspec
from test.test_utils import get_framework_and_version_from_tag, get_processor_from_image_uri


@pytest.mark.usefixtures("huggingface_only", "functionality_sanity")
@pytest.mark.processor("neuronx")
@pytest.mark.model("N/A")
@pytest.mark.integration("huggingface_training_stack")
def test_huggingface_neuronx_training_stack(training):
    _, version = get_framework_and_version_from_tag(training)
    if get_processor_from_image_uri(training) != "neuronx" or version != "2.14.0":
        pytest.skip("This suite targets the Torch NeuronX 2.14 Hugging Face image")

    root = Path(__file__).resolve().parents[3]
    buildspec = Buildspec()
    buildspec.load(root / "huggingface/pytorch/training/buildspec-2-14-neuronx.yml")
    image = buildspec["images"]["BuildNeuronHFPytorchPy314TrainingDockerImage"]
    build = {key.lower(): str(value) for key, value in image["build_args"].items()}
    build.update(
        transformers_version=str(image["transformers_version"]),
        datasets_version=str(image["datasets_version"]),
    )
    config = {
        "metadata": {
            "framework_version": str(buildspec["version"]),
            "device_type": image["device_type"],
        },
        "build": build,
    }
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "image-config.json"
        config_path.write_text(json.dumps(config))
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{root}:/workdir:ro",
            "-v",
            f"{config_path}:/image-config.json:ro",
            "--workdir",
            "/workdir",
            "-e",
            "HF_IMAGE_CONFIG=/image-config.json",
            "--entrypoint",
            "python3",
            training,
            "test/dlc_tests/container_tests/bin/huggingface/test_sanity_training.py",
        ]
        Context().run(shlex.join(command), hide=False, timeout=600)
