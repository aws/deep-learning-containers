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
"""HuggingFace PyTorch table generation."""

import re

from constants import AVAILABLE_IMAGES_TABLE_HEADER
from utils import build_ecr_url, build_public_registry_note, render_table

REPO_KEYS = [
    "huggingface-pytorch-training",
    "huggingface-pytorch-inference",
    "huggingface-pytorch-inference-neuronx",
    "huggingface-pytorch-training-neuronx",
    "huggingface-pytorch-trcomp-training",
]
DISPLAY_NAMES = {
    "huggingface-pytorch-training": "HuggingFace PyTorch Training",
    "huggingface-pytorch-inference": "HuggingFace PyTorch Inference",
    "huggingface-pytorch-inference-neuronx": "HuggingFace PyTorch Inference (NeuronX)",
    "huggingface-pytorch-training-neuronx": "HuggingFace PyTorch Training (NeuronX)",
    "huggingface-pytorch-trcomp-training": "HuggingFace PyTorch Training Compiler",
}
COLUMNS_STANDARD = [
    "Framework",
    "Python",
    "CUDA",
    "Transformers",
    "Accelerator",
    "Platform",
    "Example URL",
]
COLUMNS_NEURON = [
    "Framework",
    "Python",
    "SDK",
    "Transformers",
    "Accelerator",
    "Platform",
    "Example URL",
]


def parse_tag(tag: str) -> dict:
    """Parse HuggingFace PyTorch tag formats."""
    result = {
        "version": "",
        "transformers": "",
        "accelerator": "",
        "python": "",
        "cuda": "-",
        "sdk": "",
    }

    match = re.match(
        r"^(\d+\.\d+\.\d+)-"  # pytorch version
        r"transformers(\d+\.\d+\.\d+)-"  # transformers version
        r"(cpu|gpu|neuronx)-"  # accelerator
        r"(py\d+)-"  # python
        r"(?:(cu\d+|sdk[\d.]+)-)?",  # cuda or sdk (optional)
        tag,
    )
    if match:
        result["version"] = match.group(1)
        result["transformers"] = match.group(2)
        accel = match.group(3)
        result["accelerator"] = "NeuronX" if accel == "neuronx" else accel.upper()
        result["python"] = match.group(4)
        extra = match.group(5) or ""
        if extra.startswith("cu"):
            result["cuda"] = extra
        elif extra.startswith("sdk"):
            result["sdk"] = extra

    return result


def generate(yaml_data: dict) -> str:
    """Generate tables for HuggingFace PyTorch repositories."""
    images = yaml_data.get("images", {})
    sections = []

    for repo_key in REPO_KEYS:
        repo_data = images.get(repo_key, {})
        tags = repo_data.get("tags", [])
        if not tags:
            continue

        is_neuron = "neuron" in repo_key
        columns = COLUMNS_NEURON if is_neuron else COLUMNS_STANDARD

        rows = []
        for tag in tags:
            parsed = parse_tag(tag)
            if is_neuron:
                rows.append(
                    [
                        f"PyTorch {parsed['version']}",
                        parsed["python"],
                        parsed["sdk"],
                        parsed["transformers"],
                        parsed["accelerator"],
                        "SageMaker",
                        build_ecr_url(repo_key, tag),
                    ]
                )
            else:
                rows.append(
                    [
                        f"PyTorch {parsed['version']}",
                        parsed["python"],
                        parsed["cuda"],
                        parsed["transformers"],
                        parsed["accelerator"],
                        "SageMaker",
                        build_ecr_url(repo_key, tag),
                    ]
                )

        display_name = DISPLAY_NAMES.get(repo_key, repo_key)
        section = f"{AVAILABLE_IMAGES_TABLE_HEADER} {display_name}\n"
        if repo_data.get("public_registry"):
            section += "\n" + build_public_registry_note(repo_key) + "\n"
        section += render_table(columns, rows)
        sections.append(section)

    return "\n\n".join(sections)
