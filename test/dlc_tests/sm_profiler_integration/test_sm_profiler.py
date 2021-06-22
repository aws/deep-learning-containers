from packaging.version import Version

import pytest

from invoke.context import Context

from test.test_utils import get_framework_and_version_from_tag, get_processor_from_image_uri, get_cuda_version_from_tag, is_mainline_context


@pytest.mark.skipif(not is_mainline_context(), reason="Mainline only test")
def test_sm_profiler(training):
    tf_version_cutoff = "2.3.1"
    pt_version_cutoff = "1.6.0"
    cuda_cutoff = ""
    fw, fw_version = get_framework_and_version_from_tag(training)
    if fw not in ("tensorflow", "pytorch"):
        pytest.skip(f"Skipping SM Profiler integration test on unsupported framework {fw}")
    else:
        if fw == "tensorflow" and Version(fw_version) < Version("2.3.1"):
            pytest.skip("")

    processor = get_processor_from_image_uri(training)
    if processor not in ("cpu", "gpu"):
        pytest.skip("")



