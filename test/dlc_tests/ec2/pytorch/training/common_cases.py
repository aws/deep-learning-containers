import os

from packaging.version import Version
from packaging.specifiers import SpecifierSet

import test.test_utils.ec2 as ec2_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_framework_and_version_from_tag,
    get_cuda_version_from_tag,
    login_to_ecr_registry,
    get_account_id_from_image_uri,
)
from test.test_utils.ec2 import (
    execute_ec2_training_test,
    get_ec2_instance_type,
    get_efa_ec2_instance_type,
)

# Test functions
PT_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchStandalone")
PT_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
PT_REGRESSION_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegression")
PT_REGRESSION_CMD_REVISED = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegressionRevised"
)
PT_TORCHAUDIO_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchaudio")
PT_TORCHDATA_DEV_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchdataDev")
PT_TORCHDATA_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchdata")
PT_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")
PT_TELEMETRY_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "test_pt_dlc_telemetry_test"
)
PT_COMMON_GLOO_MPI_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi")
PT_COMMON_NCCL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl")
PT_AMP_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMP")
PT_AMP_INDUCTOR_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMPwithInductor"
)
PT_CURAND_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testCurand")
PT_APEX_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testNVApex")
PT_GDRCOPY_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "gdrcopy", "test_gdrcopy.sh")
PT_TRANSFORMER_ENGINE_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "transformerengine", "testPTTransformerEngine"
)

# Instance type filters
PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")

PT_EC2_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(default="g4dn.12xlarge")

PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(default="g4dn.12xlarge")

PT_EC2_HEAVY_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=ec2_utils.filter_efa_instance_type,
)

PT_EC2_GPU_ARM64_INSTANCE_TYPE = get_ec2_instance_type(
    default="g5g.16xlarge", processor="gpu", arch_type="arm64"
)


def pytorch_standalone(pytorch_training, ec2_connection):
    """
    Test PyTorch Standalone Sanity
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_STANDALONE_CMD, container_name="pytorch_standalone"
    )


def pytorch_training_mnist(pytorch_training, ec2_connection):
    """
    Test PyTorch MNIST
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_MNIST_CMD, container_name="pytorch_mnist"
    )


def pytorch_linear_regression_gpu(pytorch_training, ec2_connection):
    """
    Test PyTorch Linear Regression with CUDA Tensor
    """
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    image_cuda_version = get_cuda_version_from_tag(pytorch_training)
    if Version(image_framework_version) in SpecifierSet(">=2.0") and image_cuda_version >= "cu121":
        execute_ec2_training_test(
            ec2_connection,
            pytorch_training,
            PT_REGRESSION_CMD_REVISED,
            container_name="pytorch_regression_gpu",
        )
    else:
        execute_ec2_training_test(
            ec2_connection,
            pytorch_training,
            PT_REGRESSION_CMD,
            container_name="pytorch_regression_gpu",
        )


def pytorch_linear_regression_cpu(pytorch_training, ec2_connection):
    """
    Test PyTorch Linear Regression with CPU Tensor
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_REGRESSION_CMD, container_name="pytorch_regression_cpu"
    )


def pytorch_training_torchaudio(pytorch_training, ec2_connection):
    """
    Test torchaudio
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_TORCHAUDIO_CMD, container_name="pytorch_torchaudio"
    )


def pytorch_training_torchdata(pytorch_training, ec2_connection):
    """
    Test torchdata
    """
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    # HACK including PT 1.13 in this condition because the Torchdata 0.5.0 tag includes old tests data
    if Version(image_framework_version) in SpecifierSet(">=1.11,<=1.13.1"):
        execute_ec2_training_test(
            ec2_connection,
            pytorch_training,
            PT_TORCHDATA_DEV_CMD,
            container_name="pytorch_torchdata",
        )
    else:
        execute_ec2_training_test(
            ec2_connection, pytorch_training, PT_TORCHDATA_CMD, container_name="pytorch_torchdata"
        )


def pytorch_training_dgl(pytorch_training, ec2_connection):
    """
    Test DGL Package
    """
    # DGL cpu ec2 test doesn't work on PT 1.10 DLC
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_DGL_CMD, container_name="pytorch_dgl"
    )


def pytorch_telemetry_cpu(pytorch_training, ec2_connection):
    """
    Test Telemetry
    """
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        PT_TELEMETRY_CMD,
        timeout=900,
        container_name="pytorch_telemetry",
    )


def pytorch_gloo(pytorch_training, ec2_connection):
    """
    Test GLOO Backend
    """
    test_cmd = f"{PT_COMMON_GLOO_MPI_CMD} gloo 0"  # input: backend, inductor flags
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        test_cmd,
        container_name="pytorch_gloo",
        large_shm=True,
        timeout=1500,
    )


def pytorch_gloo_inductor_gpu(pytorch_training, ec2_connection):
    """
    Test GLOO Backend with PyTorch Inductor
    """
    test_cmd = f"{PT_COMMON_GLOO_MPI_CMD} gloo 1"  # input: backend, inductor flags
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        test_cmd,
        container_name="pytorch_gloo_inductor",
        large_shm=True,
        timeout=1500,
    )


def pytorch_mpi(
    pytorch_training,
    ec2_connection,
):
    """
    Test MPI Backend
    """
    test_cmd = f"{PT_COMMON_GLOO_MPI_CMD} mpi 0"  # input: backend, inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="pytorch_mpi_gloo"
    )


def pytorch_mpi_inductor_gpu(pytorch_training, ec2_connection):
    """
    Test MPI Backend with PyTorch Inductor
    """
    test_cmd = f"{PT_COMMON_GLOO_MPI_CMD} mpi 1"  # input: backend, inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="pytorch_mpi_gloo_inductor"
    )


def pytorch_nccl(pytorch_training, ec2_connection):
    """
    Test NCCL Backend
    """
    test_cmd = f"{PT_COMMON_NCCL_CMD} 0"  # input: inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="pytorch_nccl", large_shm=True
    )


def pytorch_nccl_inductor(pytorch_training, ec2_connection):
    """
    Test NCCL Backend with PyTorch Inductor
    """
    test_cmd = f"{PT_COMMON_NCCL_CMD} 1"  # input: inductor flags
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        test_cmd,
        container_name="pytorch_nccl_inductor",
        large_shm=True,
    )


def pytorch_amp(pytorch_training, ec2_connection):
    """
    Test CUDA AMP
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_AMP_CMD, container_name="pytorch_amp", timeout=1500
    )


def pytorch_amp_inductor(pytorch_training, ec2_connection):
    """
    Test CUDA AMP with PyTorch Inductor
    """
    # Native AMP was introduced in PyTorch 1.6
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        PT_AMP_INDUCTOR_CMD,
        container_name="pytorch_amp_inductor",
        timeout=1500,
    )


def pytorch_cudnn_match_gpu(pytorch_training, ec2_connection, region):
    """
    Test cuDNN Package
    PT 2.1 reintroduces a dependency on CUDNN to support NVDA TransformerEngine. This test is to ensure that torch CUDNN matches system CUDNN in the container.
    """
    container_name = "pytorch_cudnn"
    account_id = get_account_id_from_image_uri(pytorch_training)
    login_to_ecr_registry(ec2_connection, account_id, region)
    ec2_connection.run(f"docker pull -q {pytorch_training}", hide=True)
    ec2_connection.run(
        f"docker run --runtime=nvidia --gpus all --name {container_name} -itd {pytorch_training}",
        hide=True,
    )
    major_cmd = 'cat /usr/include/cudnn_version.h | grep "#define CUDNN_MAJOR"'
    minor_cmd = 'cat /usr/include/cudnn_version.h | grep "#define CUDNN_MINOR"'
    patch_cmd = 'cat /usr/include/cudnn_version.h | grep "#define CUDNN_PATCHLEVEL"'
    major = ec2_connection.run(
        f"docker exec --user root {container_name} bash -c '{major_cmd}'", hide=True
    ).stdout.split()[-1]
    minor = ec2_connection.run(
        f"docker exec --user root {container_name} bash -c '{minor_cmd}'", hide=True
    ).stdout.split()[-1]
    patch = ec2_connection.run(
        f"docker exec --user root {container_name} bash -c '{patch_cmd}'", hide=True
    ).stdout.split()[-1]

    cudnn_from_torch = ec2_connection.run(
        f"docker exec --user root {container_name} python -c 'from torch.backends import cudnn; print(cudnn.version())'",
        hide=True,
    ).stdout.strip()

    if int(major) >= 9:
        system_cudnn = f"{(int(major)*10000)+(int(minor)*100)+(int(patch))}"
    else:
        system_cudnn = f"{(int(major)*1000)+(int(minor)*100)+(int(patch))}"

    assert (
        system_cudnn == cudnn_from_torch
    ), f"System CUDNN {system_cudnn} and torch cudnn {cudnn_from_torch} do not match. Please downgrade system CUDNN or recompile torch with correct CUDNN verson."


def pytorch_curand_gpu(pytorch_training, ec2_connection):
    """
    Test cuRAND Package
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_CURAND_CMD, container_name="pytorch_curand"
    )


def pytorch_nvapex(pytorch_training, ec2_connection):
    """
    Test Nvidia Apex
    """
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_APEX_CMD, container_name="pytorch_nvapex"
    )


def pytorch_gdrcopy(pytorch_training, ec2_connection):
    """
    Test GDRCopy
    """
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        PT_GDRCOPY_CMD,
        container_name="pytorch_gdrcopy",
        enable_gdrcopy=True,
    )


def pytorch_transformer_engine(pytorch_training, ec2_connection):
    """
    Test TransformerEngine
    """
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        PT_TRANSFORMER_ENGINE_CMD,
        container_name="pytorch_transformer_engine",
    )
