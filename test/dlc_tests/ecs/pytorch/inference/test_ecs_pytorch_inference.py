import pytest

import test.test_utils.ecs as ecs_utils
import test.test_utils.ec2 as ec2_utils
from test.test_utils import request_pytorch_inference_densenet
from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2


@pytest.mark.parametrize("ecs_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_cpu(pytorch_inference, ecs_container_instance, region, cpu_only):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    ecs_cluster_name = ecs_utils.get_ecs_cluster_name(ecs_cluster_arn, region=region)
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference, "pytorch", ecs_cluster_name, model_name, worker_instance_id, region=region
        )
        inference_result = request_pytorch_inference_densenet(public_ip_address)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)


@pytest.mark.parametrize("ecs_instance_type", ["p3.2xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_pytorch_inference_gpu(pytorch_inference, ecs_container_instance, region, gpu_only):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    ecs_cluster_name = ecs_utils.get_ecs_cluster_name(ecs_cluster_arn, region=region)
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_gpus = ec2_utils.get_instance_num_gpus(worker_instance_id, region=region)

    model_name = "pytorch-densenet"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            pytorch_inference, "pytorch", ecs_cluster_name, model_name, worker_instance_id, num_gpus=num_gpus,
            region=region
        )
        inference_result = request_pytorch_inference_densenet(public_ip_address)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)
