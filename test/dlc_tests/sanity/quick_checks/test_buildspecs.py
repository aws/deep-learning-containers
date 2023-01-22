import pytest


@pytest.mark.quick_checks
@pytest.mark.model("N/A")
@pytest.mark.integration("buildspecs")
def test_train_inference_buildspec():
    """
    Walk through all buildspecs. Ensure the following:
    1. "training" or "inference is in the path
    2. if "training" is in the path, make sure "inference" is not in the file
    3. if "inference" is in the path, make sure "training" is not in the file
    """
    pass
