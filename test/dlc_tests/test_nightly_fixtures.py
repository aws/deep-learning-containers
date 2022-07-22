import pytest

# use case 1:
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_1(inference):
    print(f"\nSingle nightly fixture SMDEBUG with INFERENCE image fixture")
    pass


# use case 2:
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_2(training):
    print(f"\nSingle nightly fixture SMDEBUG with TRAINING image fixture")
    pass

# use case 3:
@pytest.mark.usefixtures("feature_smdp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_3(inference):
    print(f"\nSingle nightly fixture SMDDP with INFERENCE image fixture")
    pass

# use case 4:
@pytest.mark.usefixtures("feature_smdp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_4(training):
    print(f"\nSingle nightly fixture SMDDP with TRAINING image fixture")
    pass

# use case 5
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("feature_smdp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_5(inference):
    print(f"\nMultiple nightly fixtures SMDEBUG and SMDDP with INFERENCE image fixture")
    pass

# use case 6
@pytest.mark.usefixtures("feature_smdebug_present")
@pytest.mark.usefixtures("feature_smdp_present")
@pytest.mark.usefixtures("sagemaker")
def test_training_case_6(training):
    print(f"\nMultiple nightly fixtures SMDEBUG and SMDDP with TRAINING image fixture")
    pass


# use case 7
@pytest.mark.usefixtures("feature_smmp_present")
@pytest.mark.usefixtures("sagemaker_only")
def test_training_case_7(training):
    print(f"\nSingle nightly fixture SMMP with TRAINING image fixture")
    pass

# use case 8
@pytest.mark.usefixtures("sagemaker")
def test_training_case_8(training):
    print(f"\nNo nightly fixture with TRAINING image fixture")
    pass

