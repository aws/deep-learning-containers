import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from packaging.version import Version
from packaging.specifiers import SpecifierSet
import pytest

import test.test_utils as test_utils
import test.test_utils.ec2 as ec2_utils

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    UBUNTU_18_HPU_DLAMI_US_WEST_2,
    get_framework_and_version_from_tag,
    get_cuda_version_from_tag,
)
from test.test_utils.ec2 import (
    execute_ec2_training_test,
    get_ec2_instance_type,
    get_efa_ec2_instance_type,
)


PT_STANDALONE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchStandalone")
PT_MNIST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
PT_REGRESSION_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegression")
PT_REGRESSION_CMD_REVISED = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchRegressionRevised"
)
PT_DCGM_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "dcgm_test.sh")
PT_NCCL_LOCAL_TEST_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "healthcheck_tests", "nccl_test.sh")
PT_DGL_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "dgl_tests", "testPyTorchDGL")
PT_APEX_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testNVApex")
PT_AMP_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMP")
PT_AMP_INDUCTOR_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchAMPwithInductor"
)
PT_TELEMETRY_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "test_pt_dlc_telemetry_test"
)
PT_S3_PLUGIN_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchS3Plugin")
PT_HABANA_TEST_SUITE_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "testHabanaPTSuite")
PT_TORCHAUDIO_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchaudio")
PT_TORCHDATA_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchdata")
PT_NEURON_ALLREDUCE_SCRIPT = os.path.join(
    CONTAINER_TESTS_PREFIX, "pytorch_tests", "testNeuronSingleAllReduce"
)
PT_NEURON_MNIST_SCRIPT = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testNeuronMlp")
PT_NEURON_ALLREDUCE_CMD = f"torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=2022 {PT_NEURON_ALLREDUCE_SCRIPT}"
PT_NEURON_MLP_CMD = f"torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=2022 {PT_NEURON_MNIST_SCRIPT}"
PT_TORCHDATA_DEV_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testTorchdataDev")

PT_INDUCTOR_TEST_INSTANCE_TYPE = get_ec2_instance_type(
    default="g4dn.12xlarge", processor="gpu", filter_function=ec2_utils.filter_non_g3_instance_type
)
PT_EC2_GPU_INSTANCE_TYPE = get_ec2_instance_type(default="g3.8xlarge", processor="gpu")
PT_EC2_MULTI_GPU_NO_G_INSTANCE_TYPE = get_ec2_instance_type(
    default="p3.8xlarge",
    processor="gpu",
    filter_function=ec2_utils.filter_only_multi_gpu_and_no_g_type,
)
PT_EC2_CPU_INSTANCE_TYPE = get_ec2_instance_type(default="c5.9xlarge", processor="cpu")
PT_EC2_SINGLE_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="p3.2xlarge",
    processor="gpu",
    filter_function=ec2_utils.filter_only_single_gpu,
)
PT_EC2_MULTI_GPU_INSTANCE_TYPE = get_ec2_instance_type(
    default="g3.8xlarge",
    processor="gpu",
    filter_function=ec2_utils.filter_only_multi_gpu,
)
PT_EC2_HPU_INSTANCE_TYPE = get_ec2_instance_type(default="dl1.24xlarge", processor="hpu")
PT_EC2_NEURON_TRN1_INSTANCE_TYPE = get_ec2_instance_type(
    default="trn1.2xlarge", processor="neuronx", job_type="training"
)
PT_EC2_NEURON_INF2_INSTANCE_TYPE = get_ec2_instance_type(
    default="inf2.xlarge", processor="neuronx", job_type="training"
)

PT_EC2_EFA_GPU_INSTANCE_TYPE_AND_REGION = get_efa_ec2_instance_type(
    default="p4d.24xlarge",
    filter_function=ec2_utils.filter_efa_instance_type,
)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("all PT 2.2 tests")
@pytest.mark.model("N/A")
@pytest.mark.team("conda")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_2_2_gpu(pytorch_training, ec2_connection, region, gpu_only, ec2_instance_type):
    # pytorch_training = pytorch_training___2__2
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )

    test_cases = [
        (_pytorch_standalone_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_train_mnist_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_linear_regression_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_gloo_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_gloo_inductor_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_nccl, (pytorch_training, ec2_connection)),
        (_pytorch_nccl_inductor, (pytorch_training, ec2_connection)),
        (_pytorch_mpi_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_mpi_inductor_gpu, (pytorch_training, ec2_connection)),
        (_nvapex, (pytorch_training, ec2_connection)),
        (_pytorch_amp, (pytorch_training, ec2_connection)),
        (_pytorch_amp_inductor, (pytorch_training, ec2_connection)),
        (_pytorch_training_torchaudio_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_training_torchdata_gpu, (pytorch_training, ec2_connection)),
        (_pytorch_cudnn_match_gpu, (pytorch_training, ec2_connection, region)),
    ]

    exceptions = []

    for fn, args in test_cases:
        try:
            fn(args)
        except Exception as e:
            exceptions.append(f"{fn.__name__} FAILED WITH {type(e).__name__}:\n{e}")

    assert not exceptions, f"Found {len(exceptions)} errors in PT 2.2 test {'\n\n'.join(exceptions)}


def _pytorch_standalone_gpu(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_STANDALONE_CMD, container_name="pt_standalone"
    )


def _pytorch_train_mnist_gpu(pytorch_training, ec2_connection):
    execute_ec2_training_test(
        ec2_connection, pytorch_training, PT_MNIST_CMD, container_name="pt_mnist"
    )


def _pytorch_linear_regression_gpu(pytorch_training, ec2_connection):
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


def _pytorch_gloo_gpu(pytorch_training, ec2_connection):
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


def _pytorch_gloo_inductor_gpu(pytorch_training, ec2_connection):
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



def _pytorch_nccl(pytorch_training, ec2_connection):
    """
    Tests nccl backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl") + " 0"
    )  # add inductor flag
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="nccl", large_shm=True
    )


def _pytorch_nccl_inductor(pytorch_training, ec2_connection):
    """
    Tests nccl backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl") + " 1"
    )  # add inductor flag
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, container_name="nccl_inductor", large_shm=True
    )


def _pytorch_mpi_gpu(
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


def _pytorch_mpi_inductor_gpu(
    pytorch_training,
    ec2_connection
):
    """
    Tests mpi backend with torch inductor
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " mpi 1"
    )  # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd, container_name="mpi_gloo_inductor")


def _nvapex(pytorch_training, ec2_connection):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_APEX_CMD, container_name="nvapex")


def _pytorch_amp(
    pytorch_training, ec2_connection
):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_AMP_CMD, container_name="pytorch_amp", timeout=1500)


def _pytorch_amp_inductor(
    pytorch_training, ec2_connection
):
    # Native AMP was introduced in PyTorch 1.6
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_AMP_INDUCTOR_CMD, container_name="pytorch_amp_inductor", timeout=1500)


def _pytorch_training_torchaudio_gpu(
    pytorch_training, ec2_connection
):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHAUDIO_CMD, container_name="torchaudio")


def _pytorch_training_torchdata_gpu(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type, pt111_and_above_only
):
   execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_CMD, container_name="torchdata")


def _pytorch_cudnn_match_gpu(
    pytorch_training, ec2_connection, region
):
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
