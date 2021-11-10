import pytest

import test.test_utils.ecs as ecs_utils
import test.test_utils.ec2 as ec2_utils
from test.test_utils import request_pytorch_inference_densenet, get_framework_and_version_from_tag, get_inference_server_type
from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2, ECS_AML2_GRAVITON_CPU_USWEST2

@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region, cpu_only):
    __ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region)


@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["c6g.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GRAVITON_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_graviton_cpu(pytorch_inference, ecs_container_instance, region, graviton_only):
    __ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region)


def __ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference, "pytorch", ecs_cluster_arn, model_name, worker_instance_id, region=region
        )
        server_type = get_inference_server_type(pytorch_inference)
        inference_result = request_pytorch_inference_densenet(public_ip_address, server_type=server_type)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)


@pytest.mark.integration("elastic_inference")
@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
@pytest.mark.parametrize("ei_accelerator_type", ["eia1.large"], indirect=True)
def test_ecs_pytorch_inference_eia(
        pytorch_inference_eia, ecs_container_instance, ei_accelerator_type, region, pt14_and_above_only
):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    image_framework, image_framework_version = get_framework_and_version_from_tag(pytorch_inference_eia)
    if image_framework_version == "1.3.1":
        model_name = "pytorch-densenet-v1-3-1"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference_eia, "pytorch", ecs_cluster_arn, model_name, worker_instance_id, ei_accelerator_type, region=region
        )
        server_type = get_inference_server_type(pytorch_inference_eia)
        inference_result = request_pytorch_inference_densenet(public_ip_address, model_name=model_name, server_type=server_type)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)



@pytest.mark.model("densenet")
@pytest.mark.parametrize("ecs_instance_type", ["p3.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_gpu(pytorch_inference, ecs_container_instance, region, gpu_only):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_gpus = ec2_utils.get_instance_num_gpus(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference, "pytorch", ecs_cluster_arn, model_name, worker_instance_id, num_gpus=num_gpus,
            region=region
        )
        server_type = get_inference_server_type(pytorch_inference)
        inference_result = request_pytorch_inference_densenet(public_ip_address, server_type=server_type)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)
