import pytest

import test.test_utils as test_utils

from test.test_utils.ec2 import execute_ec2_training_test

from test.dlc_tests.ec2.tensorflow.training import common_cases
from test.dlc_tests.ec2 import smclarify_cases


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("TF_general")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", common_cases.TF_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_tf_gpu(tensorflow_training, ec2_connection, region, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(tensorflow_training, ec2_instance_type):
        pytest.skip(
            f"Image {tensorflow_training} is incompatible with instance type {ec2_instance_type}"
        )

    test_cases = [
        (common_cases.tensorflow_standalone, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_mnist, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_opencv, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_telemetry, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_tensorboard, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_addons, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_io_s3_plugin, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_dataservice, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_distribute_dataservice, (tensorflow_training, ec2_connection)),
    ]

    if "sagemaker" in tensorflow_training:
        test_cases.append(
            (smclarify_cases.smclarify_metrics_gpu, (tensorflow_training, ec2_connection)),
        )

    test_utils.execute_serial_test_cases(test_cases, test_description="TF GPU")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("TF_general")
@pytest.mark.model("N/A")
@pytest.mark.team("frameworks")
@pytest.mark.parametrize("ec2_instance_type", common_cases.TF_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_tf_cpu(tensorflow_training, ec2_connection, cpu_only):
    test_cases = [
        (common_cases.tensorflow_standalone, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_mnist, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_opencv, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_telemetry, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_tensorboard, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_addons, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_io_s3_plugin, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_dataservice, (tensorflow_training, ec2_connection)),
        (common_cases.tensorflow_distribute_dataservice, (tensorflow_training, ec2_connection)),
    ]

    if "sagemaker" in tensorflow_training:
        test_cases.append(
            (smclarify_cases.smclarify_metrics_cpu, (tensorflow_training, ec2_connection))
        )

    test_utils.execute_serial_test_cases(test_cases, test_description="TF CPU")


@pytest.mark.integration("tensorflow-dataservice-distribute-test")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", common_cases.TF_EC2_HPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True
)
def test_tensorflow_standalone_hpu(
    tensorflow_training_habana, ec2_connection, upload_habana_test_artifact
):
    execute_ec2_training_test(
        ec2_connection,
        tensorflow_training_habana,
        common_cases.TF_HABANA_TEST_SUITE_CMD,
        container_name="ec2_training_habana_tensorflow_container",
        enable_habana_async_execution=True,
    )
