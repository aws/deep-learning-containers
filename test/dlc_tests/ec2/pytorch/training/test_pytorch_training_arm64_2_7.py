import pytest

import test.test_utils as test_utils

from test.test_utils import ec2

from test.dlc_tests.ec2.pytorch.training import common_cases
from test.dlc_tests.ec2 import smclarify_cases


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pytorch_gpu_tests")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.parametrize(
    "ec2_instance_type", common_cases.PT_EC2_GPU_ARM64_INSTANCE_TYPE, indirect=True
)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_pytorch_2_7_gpu(
    pytorch_training_arm64___2__7, ec2_connection, region, gpu_only, ec2_instance_type
):
    pytorch_training = pytorch_training_arm64___2__7

    test_cases = [
        (common_cases.pytorch_standalone, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_mnist, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_linear_regression_gpu, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_nccl, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_torchaudio, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_torchdata, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_cudnn_match_gpu, (pytorch_training, ec2_connection, region)),
        (common_cases.pytorch_curand_gpu, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_telemetry_framework_gpu, (pytorch_training, ec2_connection)),
    ]

    if "sagemaker" in pytorch_training:
        test_cases.append(
            (smclarify_cases.smclarify_metrics_gpu, (pytorch_training, ec2_connection)),
        )

    # AMP must be run on multi_gpu
    if ec2.is_instance_multi_gpu(ec2_instance_type):
        test_cases.append((common_cases.pytorch_amp, (pytorch_training, ec2_connection)))

    test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.7 GPU")


# @pytest.mark.usefixtures("sagemaker")
# @pytest.mark.integration("pytorch_gpu_heavy_tests")
# @pytest.mark.model("N/A")
# @pytest.mark.team("conda")
# @pytest.mark.parametrize(
#     "ec2_instance_type", common_cases.PT_EC2_HEAVY_GPU_ARM64_INSTANCE_TYPE, indirect=True
# )
# @pytest.mark.parametrize(
#     "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
# )
# @pytest.mark.skipif(
#     test_utils.is_pr_context() and not ec2.are_heavy_instance_ec2_tests_enabled(),
#     reason="Skip GPU Heavy tests in PR context unless explicitly enabled",
# )
# def test_pytorch_2_7_gpu_heavy(
#     pytorch_training_arm64___2__7, ec2_connection, region, gpu_only, ec2_instance_type
# ):
#     pytorch_training = pytorch_training_arm64___2__7

#     test_cases = [
#         (common_cases.pytorch_gdrcopy, (pytorch_training, ec2_connection)),
#         (common_cases.pytorch_transformer_engine, (pytorch_training, ec2_connection)),
#     ]

#     test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.7 GPU Heavy")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("inductor")
@pytest.mark.model("N/A")
@pytest.mark.team("training-compiler")
@pytest.mark.parametrize(
    "ec2_instance_type", common_cases.PT_EC2_GPU_ARM64_INSTANCE_TYPE, indirect=True
)
@pytest.mark.parametrize(
    "ec2_instance_ami", [test_utils.AL2023_BASE_DLAMI_ARM64_US_WEST_2], indirect=True
)
def test_pytorch_2_7_gpu_inductor(
    pytorch_training_arm64___2__7, ec2_connection, region, gpu_only, ec2_instance_type
):
    pytorch_training = pytorch_training_arm64___2__7

    test_cases = [
        (common_cases.pytorch_nccl_inductor, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_amp_inductor, (pytorch_training, ec2_connection)),
    ]

    test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.7 GPU Inductor")
