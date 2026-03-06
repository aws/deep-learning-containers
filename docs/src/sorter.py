"""Sorting tiebreaker functions for image tables."""

import re

from constants import GLOBAL_CONFIG
from utils import parse_version


def repository_sorter(img) -> int:
    """Repository order: by table_order index."""
    table_order = GLOBAL_CONFIG["table_order"]
    try:
        return table_order.index(img.repository)
    except ValueError:
        return len(table_order)


def platform_sorter(img) -> int:
    """Platform order: SageMaker before EC2."""
    return 0 if img.get("platform") == "sagemaker" else 1


def accelerator_sorter(img) -> int:
    """Accelerator order: GPU before NeuronX before CPU."""
    return {"gpu": 0, "neuronx": 1, "cpu": 2}.get(img.get("accelerator", "").lower(), 3)


def engine_sorter(img) -> tuple:
    """Engine order for DJL: LMI before TensorRT-LLM before None, then by version desc."""
    engine = img.get("engine", "")
    version_match = re.search(r"(\d+(?:\.\d+)*)", engine)
    version = parse_version(version_match.group(1)) if version_match else parse_version("0")
    engine_lower = engine.lower()
    if "lmi" in engine_lower:
        return (0, -version.major, -version.minor, -version.micro)
    if "tensorrt" in engine_lower:
        return (1, -version.major, -version.minor, -version.micro)
    return (2, 0, 0, 0)
