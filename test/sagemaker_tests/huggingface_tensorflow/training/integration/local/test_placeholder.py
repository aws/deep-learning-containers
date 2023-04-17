import pytest
import os


@pytest.mark.processor("gpu")
@pytest.mark.model("placeholder")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
# TODO: Remove huggingface-tensorflow-training skip condition from testrunner before adding tests
def test_placeholder():
    pass
