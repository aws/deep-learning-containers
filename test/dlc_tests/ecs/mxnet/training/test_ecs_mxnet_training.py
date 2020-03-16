import datetime
from invoke import run

import test.test_utils.ecs as ecs_utils
from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2


def test_training_dummy(mxnet_training):
    print(mxnet_training)
    processor = "gpu" if "gpu" in mxnet_training else "cpu"
    framework = "mxnet"
    job = "training"
    python_version = "py2" if "py2" in mxnet_training else "py3"
    datetime_suffix = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

    s3_test_location_request = f"{framework}-{job}-{processor}-{python_version}-{datetime_suffix}"
    s3_test_location = ecs_utils.upload_tests_for_ecs(s3_test_location_request)
    assert s3_test_location, "Upload to s3 failed"

    ecs_test_command = ecs_utils.build_ecs_training_command(s3_test_location, "/test/bin/testMXNet")
    print(ecs_test_command)

    run(f"aws s3 ls --recursive {s3_test_location}/")
