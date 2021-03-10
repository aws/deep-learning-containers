import pytest
from packaging.specifiers import SpecifierSet

from test.test_utils import get_framework_and_version_from_tag, get_cuda_version_from_tag


def validate_or_skip_smmodelparallel(ecr_image):
	_, image_framework_version = get_framework_and_version_from_tag(ecr_image)
    image_cuda_version = get_cuda_version_from_tag(ecr_image)
    if not(Version(image_framework_version) in SpecifierSet(">=1.6,<1.9")) or image_cuda_version != "cu110":
        pytest.skip("Model Parallelism only supports CUDA 11 on PyTorch 1.6, 1.7 and 1.8")
