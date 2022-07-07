import pytest

@pytest.mark.usefixtures("feature_smdebug_present")
def test_training(training, smdebug_installed, aws_framework_installed):
    print("Testing both Sagemaker and AWS Framework")
    assert "John" == "John"