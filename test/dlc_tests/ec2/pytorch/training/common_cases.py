import os

from packaging.version import Version
from packaging.specifiers import SpecifierSet

import test.test_utils.ec2 as ec2_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    get_framework_and_version_from_tag,
    get_cuda_version_from_tag,
)
from test.test_utils.ec2 import (
    execute_ec2_training_test,
    get_ec2_instance_type,
    get_efa_ec2_instance_type,
)

# Test functions
PT_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchStandalone")
CURAND_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testCurand")
PT_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
PT_REGRESSION_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegression")
PT_REGRESSION_CMD_REVISED = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegressionRevised"
)
PT_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")
PT_APEX_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testNVApex")
PT_AMP_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMP")
PT_AMP_INDUCTOR_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMPwithInductor"
)
PT_TELEMETRY_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "test_pt_dlc_telemetry_test"
)
PT_TORCHAUDIO_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchaudio")
PT_TORCHDATA_DEV_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchdataDev")
PT_TORCHDATA_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchdata")

# Instance type filters
PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")

PT_EC2_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(default="g4dn.12xlarge")

PT_EC2_GPU_INDUCTOR_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="g4dn.12xlarge", filter_function=ec2_utils.filter_non_g3_instance_type
)


def pytorch_standalone(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_STANDALONE_CMD, container_name="pt_standalone"
    )


def pytorch_train_mnist(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_MNIST_CMD, container_name="pt_mnist"
    )


def pytorch_linear_regression_gpu(pytorch_training, ec2_connection):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    image_cuda_version = get_cuda_version_from_tag(pytorch_training)
    if Version(image_framework_version) in SpecifierSet(">=2.0") and image_cuda_version >= "cu121":
        execute_ec2_training_test(
            ec2_connection, pytorch_training, PT_REGRESSION_CMD_REVISED, container_name="pt_reg"
        )
    else:
        execute_ec2_training_test(
            ec2_connection, pytorch_training, PT_REGRESSION_CMD, container_name="pt_reg"
        )


def pytorch_gloo(pytorch_training, ec2_connection):
    """
    Tests gloo backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " gloo 0"
    )  # backend, inductor flags
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        test_cmd,
        container_name="gloo",
        large_shm=True,
        timeout=1500,
    )


def pytorch_gloo_inductor_gpu(pytorch_training, ec2_connection):
    """
    Tests gloo backend with torch inductor
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " gloo 1"
    )  # backend, inductor flags
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        test_cmd,
        container_name="gloo_inductor",
        large_shm=True,
        timeout=1500,
    )


def pytorch_nccl(pytorch_training, ec2_connection):
    """
    Tests nccl backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl") + " 0"
    )  # add inductor flag
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="nccl", large_shm=True
    )


def pytorch_nccl_inductor(pytorch_training, ec2_connection):
    """
    Tests nccl backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl") + " 1"
    )  # add inductor flag
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="nccl_inductor", large_shm=True
    )


def pytorch_mpi(
    pytorch_training,
    ec2_connection,
):
    """
    Tests mpi backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " mpi 0"
    )  # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd, container_name="mpi_gloo")


def pytorch_mpi_inductor_gpu(pytorch_training, ec2_connection):
    """
    Tests mpi backend with torch inductor
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " mpi 1"
    )  # backend, inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="mpi_gloo_inductor"
    )


def nvapex(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_APEX_CMD, container_name="nvapex"
    )


def pytorch_amp(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_AMP_CMD, container_name="pytorch_amp", timeout=1500
    )


def pytorch_amp_inductor(pytorch_training, ec2_connection):
    # Native AMP was introduced in PyTorch 1.6
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training,
        PT_AMP_INDUCTOR_CMD,
        container_name="pytorch_amp_inductor",
        timeout=1500,
    )


def pytorch_training_torchaudio(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_TORCHAUDIO_CMD, container_name="torchaudio"
    )


def pytorch_cudnn_match_gpu(pytorch_training, ec2_connection, region):
    """
    PT 2.1 reintroduces a dependency on CUDNN to support NVDA TransformerEngine. This test is to ensure that torch CUDNN matches system CUDNN in the container.
    """
    container_name = "pt_cudnn_test"
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)
    ec2_connection.run(f"docker pull -q {pytorch_training}", hide=True)
    ec2_connection.run(
        f"nvidia-docker run --name {container_name} -itd {pytorch_training}", hide=True
    )
    major_cmd = 'cat /usr/include/cudnn_version.h | grep "#define CUDNN_MAJOR"'
    minor_cmd = 'cat /usr/include/cudnn_version.h | grep "#define CUDNN_MINOR"'
    patch_cmd = 'cat /usr/include/cudnn_version.h | grep "#define CUDNN_PATCHLEVEL"'
    major = ec2_connection.run(
        f"nvidia-docker exec --user root {container_name} bash -c '{major_cmd}'", hide=True
    ).stdout.split()[-1]
    minor = ec2_connection.run(
        f"nvidia-docker exec --user root {container_name} bash -c '{minor_cmd}'", hide=True
    ).stdout.split()[-1]
    patch = ec2_connection.run(
        f"nvidia-docker exec --user root {container_name} bash -c '{patch_cmd}'", hide=True
    ).stdout.split()[-1]

    cudnn_from_torch = ec2_connection.run(
        f"nvidia-docker exec --user root {container_name} python -c 'from torch.backends import cudnn; print(cudnn.version())'",
        hide=True,
    ).stdout.strip()

    if len(patch) == 1:
        patch = f"0{patch}"

    system_cudnn = f"{major}{minor}{patch}"
    assert (
        system_cudnn == cudnn_from_torch
    ), f"System CUDNN {system_cudnn} and torch cudnn {cudnn_from_torch} do not match. Please downgrade system CUDNN or recompile torch with correct CUDNN verson."


def pytorch_curand_gpu(pytorch_training, ec2_connection):
    execute_ec2_training_test(ec2_connection, pytorch_training, CURAND_CMD)


def pytorch_linear_regression_cpu(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_REGRESSION_CMD, container_name="pt_reg"
    )


def pytorch_train_dgl_cpu(pytorch_training, ec2_connection):
    # DGL cpu ec2 test doesn't work on PT 1.10 DLC
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_DGL_CMD, container_name="dgl_cpu"
    )


def pytorch_telemetry_cpu(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_TELEMETRY_CMD, timeout=900, container_name="telemetry"
    )


def pytorch_training_torchdata(pytorch_training, ec2_connection):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    # HACK including PT 1.13 in this condition because the Torchdata 0.5.0 tag includes old tests data
    if Version(image_framework_version) in SpecifierSet(">=1.11,<=1.13.1"):
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_DEV_CMD)
    else:
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_CMD)
