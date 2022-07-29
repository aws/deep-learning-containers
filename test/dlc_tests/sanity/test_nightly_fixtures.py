import pytest

from test.test_utils.imageutils import (
    get_image_labels
)

# use case 1:
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_1(inference):
    pass


# use case 2:
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_2(training):
    pass

# use case 3:
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_3(inference):
    pass

# use case 4:
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_4(training):
    pass

# use case 5
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_5(inference):
    pass

# use case 6
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("feature_smddp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_6(training):
    pass


# use case 7
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.usefixtures("sagemaker_only")
def test_training_case_7(training):
    pass

# use case 8
@pytest.mark.usefixtures("sagemaker")
def test_training_case_8(training):
    pass