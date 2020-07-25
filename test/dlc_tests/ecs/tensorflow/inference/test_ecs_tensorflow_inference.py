import pytest

import test.test_utils.ecs as ecs_utils
import test.test_utils.ec2 as ec2_utils
from test.test_utils import get_tensorflow_model_name, request_tensorflow_inference
from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2


@pytest.mark.model("half_plus_two")
@pytest.mark.parametrize("ecs_instance_type", ["c5.4xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_CPU_USWEST2], indirect=True)
def test_ecs_tensorflow_inference_cpu(tensorflow_inference, ecs_container_instance, region, cpu_only):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)

    model_name = "saved_model_half_plus_two"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            tensorflow_inference, "tensorflow", ecs_cluster_arn, model_name, worker_instance_id, region=region
        )
        model_name = get_tensorflow_model_name("cpu", model_name)
        inference_result = request_tensorflow_inference(model_name, ip_address=public_ip_address)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)


@pytest.mark.model("half_plus_two")
@pytest.mark.parametrize("ecs_instance_type", ["p2.8xlarge"], indirect=True)
@pytest.mark.parametrize("ecs_ami", [ECS_AML2_GPU_USWEST2], indirect=True)
def test_ecs_tensorflow_inference_gpu(tensorflow_inference, ecs_container_instance, region, gpu_only):
    worker_instance_id, ecs_cluster_arn = ecs_container_instance
    public_ip_address = ec2_utils.get_public_ip(worker_instance_id, region=region)
    num_gpus = ec2_utils.get_instance_num_gpus(worker_instance_id)

    model_name = "saved_model_half_plus_two"
    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            tensorflow_inference, "tensorflow", ecs_cluster_arn, model_name, worker_instance_id, num_gpus=num_gpus,
            region=region
        )
        model_name = get_tensorflow_model_name("gpu", model_name)
        inference_result = request_tensorflow_inference(model_name, ip_address=public_ip_address)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"

    finally:
        ecs_utils.tear_down_ecs_inference_service(ecs_cluster_arn, service_name, task_family, revision)
