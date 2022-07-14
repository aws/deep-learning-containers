import pytest
import os

@pytest.mark.processor("gpu")
@pytest.mark.model("placeholder")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
#If planning to add any tests remove the huggingface-tensorflow-training from testrunner.py file
def test_placeholder():
    pass
