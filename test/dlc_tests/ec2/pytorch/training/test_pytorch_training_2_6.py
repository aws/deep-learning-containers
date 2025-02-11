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
    "ec2_instance_type, region", common_cases.PT_EC2_GPU_INSTANCE_TYPE_AND_REGION, indirect=True
)
def test_pytorch_2_6_gpu(
    pytorch_training___2__6, ec2_connection, region, gpu_only, ec2_instance_type
):
    pytorch_training = pytorch_training___2__6
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )

    test_cases = [
        (common_cases.pytorch_standalone, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_mnist, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_linear_regression_gpu, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_gloo, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_nccl, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_mpi, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_torchaudio, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_cudnn_match_gpu, (pytorch_training, ec2_connection, region)),
        (common_cases.pytorch_curand_gpu, (pytorch_training, ec2_connection)),
    ]

    if "sagemaker" in pytorch_training:
        test_cases.append(
            (smclarify_cases.smclarify_metrics_gpu, (pytorch_training, ec2_connection)),
        )

    # AMP must be run on multi_gpu
    if ec2.is_instance_multi_gpu(ec2_instance_type):
        test_cases.append((common_cases.pytorch_amp, (pytorch_training, ec2_connection)))

    test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.6 GPU")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pytorch_gpu_heavy_tests")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.parametrize(
    "ec2_instance_type, region",
    common_cases.PT_EC2_HEAVY_GPU_INSTANCE_TYPE_AND_REGION,
    indirect=True,
)
@pytest.mark.skipif(
    test_utils.is_pr_context() and not ec2.are_heavy_instance_ec2_tests_enabled(),
    reason="Skip GPU Heavy tests in PR context unless explicitly enabled",
)
def test_pytorch_2_6_gpu_heavy(
    pytorch_training___2__6, ec2_connection, region, gpu_only, ec2_instance_type
):
    pytorch_training = pytorch_training___2__6
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )

    test_cases = [
        (common_cases.pytorch_gdrcopy, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_transformer_engine, (pytorch_training, ec2_connection)),
    ]

    test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.6 GPU Heavy")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("inductor")
@pytest.mark.model("N/A")
@pytest.mark.team("training-compiler")
@pytest.mark.parametrize(
    "ec2_instance_type, region",
    common_cases.PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_AND_REGION,
    indirect=True,
)
def test_pytorch_2_6_gpu_inductor(
    pytorch_training___2__6, ec2_connection, region, gpu_only, ec2_instance_type
):
    pytorch_training = pytorch_training___2__6
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )

    test_cases = [
        (common_cases.pytorch_gloo_inductor_gpu, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_mpi_inductor_gpu, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_nccl_inductor, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_amp_inductor, (pytorch_training, ec2_connection)),
    ]

    test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.6 GPU Inductor")


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pytorch_cpu_tests")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.parametrize("ec2_instance_type", common_cases.PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_2_6_cpu(pytorch_training___2__6, ec2_connection, cpu_only):
    pytorch_training = pytorch_training___2__6

    test_cases = [
        (common_cases.pytorch_standalone, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_mnist, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_linear_regression_cpu, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_gloo, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_mpi, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_training_torchaudio, (pytorch_training, ec2_connection)),
        (common_cases.pytorch_telemetry_cpu, (pytorch_training, ec2_connection)),
    ]

    if "sagemaker" in pytorch_training:
        test_cases += [
            (smclarify_cases.smclarify_metrics_cpu, (pytorch_training, ec2_connection)),
        ]

    test_utils.execute_serial_test_cases(test_cases, test_description="PT 2.6 CPU")
