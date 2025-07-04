import test.test_utils.ec2 as ec2_utils
import test.test_utils.ecs as ecs_utils
from test.test_utils import (
    AML2_BASE_ARM64_DLAMI_US_WEST_2,
    ECS_AML2_ARM64_CPU_USWEST2,
    ECS_AML2_CPU_USWEST2,
    ECS_AML2_GPU_USWEST2,
    ECS_AML2_NEURON_USWEST2,
    get_framework_and_version_from_tag,
    get_inference_server_type,
    request_pytorch_inference_densenet,
)

import pytest


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
@pytest.mark.team("conda")
def test_ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region, cpu_only):
    __ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_ARM64_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_graviton_cpu(
    pytorch_inference_graviton, ecs_container_instance, region, cpu_only
):
    __ecs_pytorch_inference_cpu(pytorch_inference_graviton, ecs_container_instance, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_ARM64_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_arm64_cpu(
    pytorch_inference_arm64, ecs_container_instance, region, cpu_only
):
    __ecs_pytorch_inference_cpu(pytorch_inference_arm64, ecs_container_instance, region)


def __ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference,
            "pytorch",
            ecs_cluster_arn,
            model_name,
            worker_instance_id,
            region=region,
        )
        server_type = get_inference_server_type(pytorch_inference)
        inference_result = request_pytorch_inference_densenet(
            public_ip_address, server_type=server_type
        )
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(
            ecs_cluster_arn, service_name, task_family, revision
        )


@pytest.mark.model("resnet")
@pytest.mark.parametrize("ecs_instance_type", ["inf1.2xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_NEURON_USWEST2], indirect=True)
@pytest.mark.team("neuron")
def test_ecs_pytorch_inference_neuron(pytorch_inference_neuron, ecs_container_instance, region):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_neurons = ec2_utils.get_instance_num_inferentias(worker_instance_id, region=region)

    model_name = "pytorch-resnet-neuron"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference_neuron,
            "pytorch",
            ecs_cluster_arn,
            model_name,
            worker_instance_id,
            num_neurons=num_neurons,
            region=region,
        )
        server_type = get_inference_server_type(pytorch_inference_neuron)
        inference_result = request_pytorch_inference_densenet(
            public_ip_address, server_type=server_type, model_name=model_name
        )
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(
            ecs_cluster_arn, service_name, task_family, revision
        )


@pytest.mark.model("resnet")
@pytest.mark.parametrize("ecs_instance_type", ["trn1.2xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_NEURON_USWEST2], indirect=True)
@pytest.mark.team("neuron")
def test_ecs_pytorch_inference_neuronx(pytorch_inference_neuronx, ecs_container_instance, region):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_neurons = 1

    model_name = "pytorch-resnet-neuronx"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference_neuronx,
            "pytorch",
            ecs_cluster_arn,
            model_name,
            worker_instance_id,
            num_neurons=num_neurons,
            region=region,
        )
        server_type = get_inference_server_type(pytorch_inference_neuronx)
        inference_result = request_pytorch_inference_densenet(
            public_ip_address, server_type=server_type, model_name=model_name
        )
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(
            ecs_cluster_arn, service_name, task_family, revision
        )


@pytest.mark.model("resnet")
@pytest.mark.parametrize("ecs_instance_type", ["inf2.xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_NEURON_USWEST2], indirect=True)
@pytest.mark.team("neuron")
def test_ecs_pytorch_inference_neuronx_inf2(
    pytorch_inference_neuronx, ecs_container_instance, region
):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_neurons = 1

    model_name = "pytorch-resnet-neuronx"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference_neuronx,
            "pytorch",
            ecs_cluster_arn,
            model_name,
            worker_instance_id,
            num_neurons=num_neurons,
            region=region,
        )
        server_type = get_inference_server_type(pytorch_inference_neuronx)
        inference_result = request_pytorch_inference_densenet(
            public_ip_address, server_type=server_type, model_name=model_name
        )
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(
            ecs_cluster_arn, service_name, task_family, revision
        )


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["g4dn.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
@pytest.mark.team("conda")
def test_ecs_pytorch_inference_gpu(pytorch_inference, ecs_container_instance, region, gpu_only):
    __ecs_pytorch_inference_gpu(pytorch_inference, ecs_container_instance, region)


@pytest.mark.skip(reason="No ECS optimized AMI available for ARM64+GPU")
@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["g5g.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [AML2_BASE_ARM64_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.team("conda")
def test_ecs_pytorch_inference_graviton_gpu(
    pytorch_inference_graviton, ecs_container_instance, region, gpu_only
):
    __ecs_pytorch_inference_gpu(pytorch_inference_graviton, ecs_container_instance, region)


@pytest.mark.skip(reason="No ECS optimized AMI available for ARM64+GPU")
@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["g5g.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [AML2_BASE_ARM64_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.team("conda")
def test_ecs_pytorch_inference_arm64_gpu(
    pytorch_inference_arm64, ecs_container_instance, region, gpu_only
):
    __ecs_pytorch_inference_gpu(pytorch_inference_arm64, ecs_container_instance, region)


def __ecs_pytorch_inference_gpu(pytorch_inference, ecs_container_instance, region):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_gpus = ec2_utils.get_instance_num_gpus(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference,
            "pytorch",
            ecs_cluster_arn,
            model_name,
            worker_instance_id,
            num_gpus=num_gpus,
            region=region,
        )
        server_type = get_inference_server_type(pytorch_inference)
        inference_result = request_pytorch_inference_densenet(
            public_ip_address, server_type=server_type
        )
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(
            ecs_cluster_arn, service_name, task_family, revision
        )
