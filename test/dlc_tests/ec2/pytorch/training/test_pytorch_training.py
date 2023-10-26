import os

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
PT_BERT_INDUCTOR_CMD = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorch")
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


@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL20_PT_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_NEURON_TRN1_INSTANCE_TYPE, indirect=True)
@pytest.mark.integration("pytorch_neuronx_sanity_test")
@pytest.mark.neuronx_test
@pytest.mark.model("xla")
def test_pytorch_allreduce_neuronx(pytorch_training_neuronx, ec2_connection):
    execute_ec2_training_test(ec2_connection, pytorch_training_neuronx, PT_NEURON_ALLREDUCE_CMD)


@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL20_PT_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_NEURON_TRN1_INSTANCE_TYPE, indirect=True)
@pytest.mark.integration("pytorch_neuronx_sanity_test")
@pytest.mark.neuronx_test
@pytest.mark.model("mlp")
def test_pytorch_train_mlp_neuronx(pytorch_training_neuronx, ec2_connection):
    execute_ec2_training_test(ec2_connection, pytorch_training_neuronx, PT_NEURON_MLP_CMD)


@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL20_PT_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_NEURON_INF2_INSTANCE_TYPE, indirect=True)
@pytest.mark.integration("pytorch_neuronx_sanity_test")
@pytest.mark.neuronx_test
@pytest.mark.model("xla")
def test_pytorch_allreduce_neuronx_inf2(pytorch_training_neuronx, ec2_connection):
    execute_ec2_training_test(ec2_connection, pytorch_training_neuronx, PT_NEURON_ALLREDUCE_CMD)


@pytest.mark.parametrize("ec2_instance_ami", [test_utils.UL20_PT_NEURON_US_WEST_2], indirect=True)
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_NEURON_INF2_INSTANCE_TYPE, indirect=True)
@pytest.mark.integration("pytorch_neuronx_sanity_test")
@pytest.mark.neuronx_test
@pytest.mark.model("mlp")
def test_pytorch_train_mlp_neuronx_inf2(pytorch_training_neuronx, ec2_connection):
    execute_ec2_training_test(ec2_connection, pytorch_training_neuronx, PT_NEURON_MLP_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pytorch_sanity_test")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_standalone_gpu(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_STANDALONE_CMD)


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.integration("pytorch_sanity_healthcheck_test")
@pytest.mark.model("N/A")
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_healthcheck_dcgm(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DCGM_TEST_CMD)


@pytest.mark.usefixtures("sagemaker_only")
@pytest.mark.usefixtures("pt201_and_above_only")
@pytest.mark.integration("pytorch_sanity_healthcheck_test")
@pytest.mark.model("N/A")
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_MULTI_GPU_NO_G_INSTANCE_TYPE, indirect=True)
def test_pytorch_healthcheck_nccl(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_NCCL_LOCAL_TEST_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pytorch_sanity_test")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_standalone_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_STANDALONE_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_mnist_gpu(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("bert")
@pytest.mark.integration("inductor")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_bert_gpu(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_BERT_INDUCTOR_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("mnist")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_mnist_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_MNIST_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("linear_regression")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_linear_regression_gpu(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    image_cuda_version = get_cuda_version_from_tag(pytorch_training)
    if Version(image_framework_version) in SpecifierSet(">=2.0") and image_cuda_version >= "cu121":
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD_REVISED)
    else:
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("linear_regression")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_linear_regression_cpu(pytorch_training, ec2_connection, cpu_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_REGRESSION_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.skip_pt20_cuda121_tests
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_dgl_gpu(
    pytorch_training, ec2_connection, ec2_instance_type, gpu_only, py3_only, skip_pt110
):
    # DGL gpu ec2 test doesn't work on PT 1.10 DLC
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    image_cuda_version = get_cuda_version_from_tag(pytorch_training)
    # TODO: Remove when DGL gpu test on ec2 get fixed
    if (
        Version(image_framework_version) in SpecifierSet("==1.10.*")
        and image_cuda_version == "cu113"
    ):
        pytest.skip("ecs test for DGL gpu fails for pt 1.10")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DGL_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("dgl")
@pytest.mark.model("gcn")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_train_dgl_cpu(pytorch_training, ec2_connection, cpu_only, py3_only, skip_pt110):
    # DGL cpu ec2 test doesn't work on PT 1.10 DLC
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_DGL_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("gloo")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_INDUCTOR_TEST_INSTANCE_TYPE, indirect=True)
def test_pytorch_gloo_gpu(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    """
    Tests gloo backend
    """
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " gloo 0"
    )  # backend, inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, large_shm=True, timeout=1500
    )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("gloo")
@pytest.mark.integration("inductor")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_INDUCTOR_TEST_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_inductor_test
def test_pytorch_gloo_inductor_gpu(
    pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type
):
    """
    Tests gloo backend with torch inductor
    """
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " gloo 1"
    )  # backend, inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, large_shm=True, timeout=1500
    )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("gloo")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_gloo_cpu(pytorch_training, ec2_connection, cpu_only, py3_only, ec2_instance_type):
    """
    Tests gloo backend
    """
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " gloo 0"
    )  # backend, inductor flags
    execute_ec2_training_test(
        ec2_connection, pytorch_training, test_cmd, large_shm=True, timeout=1500
    )


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("nccl")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_INDUCTOR_TEST_INSTANCE_TYPE, indirect=True)
def test_pytorch_nccl(pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type):
    """
    Tests nccl backend
    """
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl") + " 0"
    )  # add inductor flag
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd, large_shm=True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("nccl")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_INDUCTOR_TEST_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_inductor_test
def test_pytorch_nccl_inductor(
    pytorch_training, ec2_connection, gpu_only, py3_only, ec2_instance_type
):
    """
    Tests nccl backend
    """
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNccl") + " 1"
    )  # add inductor flag
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd, large_shm=True)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("nccl")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_trcomp_containers
def test_pytorch_nccl_version(
    pytorch_training,
    ec2_connection,
    ec2_instance_type,
    gpu_only,
    py3_only,
    pt17_and_above_only,
    pt200_and_below_only,
):
    """
    Tests nccl version
    """
    if "trcomp" in pytorch_training:
        pytest.skip(
            f"Image {pytorch_training} should use the system nccl through xla. Hence the test is skipped."
        )
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchNcclVersion")
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("mpi")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_mpi_gpu(
    pytorch_training,
    ec2_connection,
    gpu_only,
    py3_only,
    ec2_instance_type,
    pt111_and_above_only,
    skip_pt200,  # PT2.0.0 doesn't support MPI https://github.com/pytorch/pytorch/issues/97507
):
    """
    Tests mpi backend
    """
    if "trcomp" in pytorch_training:
        pytest.skip(f"Image {pytorch_training} is incompatible with distribution type MPI.")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " mpi 0"
    )  # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("mpi")
@pytest.mark.integration("inductor")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_INDUCTOR_TEST_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_inductor_test
def test_pytorch_mpi_inductor_gpu(
    pytorch_training,
    ec2_connection,
    gpu_only,
    py3_only,
    ec2_instance_type,
    pt111_and_above_only,
    skip_pt200,  # PT2.0.0 doesn't support MPI https://github.com/pytorch/pytorch/issues/97507
):
    """
    Tests mpi backend with torch inductor
    """
    if "trcomp" in pytorch_training:
        pytest.skip(f"Image {pytorch_training} is incompatible with distribution type MPI.")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " mpi 1"
    )  # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("mpi")
@pytest.mark.model("resnet18")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_mpi_cpu(
    pytorch_training,
    ec2_connection,
    cpu_only,
    py3_only,
    ec2_instance_type,
    pt111_and_above_only,
    skip_pt200,  # PT2.0.0 doesn't support MPI https://github.com/pytorch/pytorch/issues/97507
):
    """
    Tests mpi backend
    """
    if "trcomp" in pytorch_training:
        pytest.skip(f"Image {pytorch_training} is incompatible with distribution type MPI.")
    test_cmd = (
        os.path.join(CONTAINER_TESTS_PREFIX, "pytorch_tests", "testPyTorchGlooMpi") + " mpi 0"
    )  # backend, inductor flags
    execute_ec2_training_test(ec2_connection, pytorch_training, test_cmd)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("nvidia_apex")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_nvapex(pytorch_training, ec2_connection, gpu_only, ec2_instance_type):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_APEX_CMD)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("amp")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_MULTI_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_amp(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type, pt16_and_above_only
):
    # Native AMP was introduced in PyTorch 1.6
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_AMP_CMD, timeout=1500)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("amp")
@pytest.mark.integration("inductor")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", PT_INDUCTOR_TEST_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_inductor_test
def test_pytorch_amp_inductor(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type, pt16_and_above_only
):
    # Native AMP was introduced in PyTorch 1.6
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_AMP_INDUCTOR_CMD, timeout=1500)


@pytest.mark.usefixtures("feature_s3_plugin_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pt_s3_plugin_gpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_s3plugin_test
def test_pytorch_s3_plugin_gpu(
    pytorch_training,
    ec2_connection,
    gpu_only,
    ec2_instance_type,
    pt18_and_above_only,
    below_pt113_only,  # PyTorch S3 Plugin has been deprecated for PT 1.13 and above
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if "trcomp" in pytorch_training and Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip(f"Image {pytorch_training} doesn't support s3. Hence test is skipped.")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_S3_PLUGIN_CMD)


@pytest.mark.usefixtures("feature_s3_plugin_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pt_s3_plugin_cpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.skip_s3plugin_test
def test_pytorch_s3_plugin_cpu(
    pytorch_training,
    ec2_connection,
    cpu_only,
    ec2_instance_type,
    pt18_and_above_only,
    below_pt113_only,  # PyTorch S3 Plugin has been deprecated for PT 1.13 and above
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_S3_PLUGIN_CMD)


@pytest.mark.usefixtures("feature_torchaudio_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pt_torchaudio_gpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_training_torchaudio_gpu(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type, pt111_and_above_only
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHAUDIO_CMD)


@pytest.mark.usefixtures("feature_torchaudio_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pt_torchaudio_cpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_training_torchaudio_cpu(
    pytorch_training, ec2_connection, cpu_only, ec2_instance_type, pt111_and_above_only
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHAUDIO_CMD)


@pytest.mark.usefixtures("feature_torchdata_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pt_torchdata_gpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_training_torchdata_gpu(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type, pt111_and_above_only
):
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if "trcomp" in pytorch_training and Version(image_framework_version) in SpecifierSet("<2.0"):
        pytest.skip(f"Image {pytorch_training} doesn't package torch_data. Hence test is skipped.")
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    # HACK including PT 1.13 in this condition because the Torchdata 0.5.0 tag includes old tests data
    if Version(image_framework_version) in SpecifierSet(">=1.11,<=1.13.1"):
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_DEV_CMD)
    else:
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_CMD)


@pytest.mark.usefixtures("feature_torchdata_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("pt_torchdata_cpu")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_training_torchdata_cpu(
    pytorch_training, ec2_connection, cpu_only, ec2_instance_type, pt111_and_above_only
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )
    # HACK including PT 1.13 in this condition because the Torchdata 0.5.0 tag includes old tests data
    _, image_framework_version = get_framework_and_version_from_tag(pytorch_training)
    if Version(image_framework_version) in SpecifierSet(">=1.11,<=1.13.1"):
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_DEV_CMD)
    else:
        execute_ec2_training_test(ec2_connection, pytorch_training, PT_TORCHDATA_CMD)


@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_telemetry_gpu(
    pytorch_training, ec2_connection, gpu_only, ec2_instance_type, pt15_and_above_only
):
    if test_utils.is_image_incompatible_with_instance_type(pytorch_training, ec2_instance_type):
        pytest.skip(
            f"Image {pytorch_training} is incompatible with instance type {ec2_instance_type}"
        )


@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("telemetry")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_CPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_telemetry_cpu(pytorch_training, ec2_connection, cpu_only, pt15_and_above_only):
    execute_ec2_training_test(ec2_connection, pytorch_training, PT_TELEMETRY_CMD, timeout=900)


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_HPU_INSTANCE_TYPE, indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True)
def test_pytorch_standalone_hpu(
    pytorch_training_habana, ec2_connection, upload_habana_test_artifact
):
    execute_ec2_training_test(
        ec2_connection,
        pytorch_training_habana,
        PT_HABANA_TEST_SUITE_CMD,
        container_name="ec2_training_habana_pytorch_container",
        enable_habana_async_execution=True,
    )


@pytest.mark.usefixtures("feature_aws_framework_present")
@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("cudnn")
@pytest.mark.model("N/A")
@pytest.mark.parametrize("ec2_instance_type", PT_EC2_SINGLE_GPU_INSTANCE_TYPE, indirect=True)
def test_pytorch_cudnn_match_gpu(
    pytorch_training, ec2_connection, region, gpu_only, ec2_instance_type, pt21_and_above_only
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
