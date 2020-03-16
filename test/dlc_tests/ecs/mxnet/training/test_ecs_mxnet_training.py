import datetime

from invoke import run
import test.test_utils.ecs as ecs_utils


def test_s3_artifact_copy(mxnet_training):
    print(mxnet_training)
    tag = mxnet_training.split(":")[-1]
    testname_datetime_suffix = f"mxnet-training-{tag}-{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}"
    s3_test_artifact_location = ecs_utils.upload_tests_for_ecs(testname_datetime_suffix)

    run_out = run(f"aws s3 ls --recursive {s3_test_artifact_location}/")
    assert run_out.return_code == 0, "Failed to copy test scripts"

    ecs_utils.delete_uploaded_tests_for_ecs(s3_test_artifact_location)
