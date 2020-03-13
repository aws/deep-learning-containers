import datetime
import pytest

import test.test_utils.ecs as ecs_utils
import test.test_utils.ec2 as ec2_utils
from test.test_utils import request_mxnet_inference_squeezenet
from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2


@pytest.mark.parametrize(
    "ecs_cluster_name",
    [f"mxnet-inference-cluster-{datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')}"],
    indirect=True,
)
def test_ecs_mxnet_inference(mxnet_inference, region, ecs_cluster_name, ecs_cluster):
    processor = "gpu" if "gpu" in mxnet_inference else "cpu"
    framework = "mxnet"
    job = "inference"
    cluster_arn = ecs_cluster
    worker_ami_id = ECS_AML2_GPU_USWEST2 if processor == 'gpu' else ECS_AML2_CPU_USWEST2
    worker_instance_type = 'p3.8xlarge' if processor == 'gpu' else 'c5.18xlarge'
    datetime_suffix = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
    worker_instance_id, public_ip_address = ecs_utils.attach_ecs_worker_node(
        worker_instance_type, worker_ami_id, ecs_cluster_name, cluster_arn, region=region
    )

    model_names = ["squeezenet"]
    num_cpus = ec2_utils.get_instance_num_cpus(worker_instance_id, region=region)
    num_gpus = str(ec2_utils.get_instance_num_gpus(worker_instance_id, region=region)) if processor == "gpu" else None
    # We assume that about 80% of RAM is free on the instance, since we are not directly querying it to find out
    # what the memory utilization is.
    memory = int(ec2_utils.get_instance_memory(worker_instance_id, region=region) * 0.8)

    service_name = task_family = revision = None
    try:
        service_name, task_family, revision = ecs_utils.setup_ecs_inference_service(
            mxnet_inference,
            framework,
            job,
            processor,
            ecs_cluster_name,
            cluster_arn,
            datetime_suffix,
            model_names,
            num_cpus,
            memory,
            num_gpus,
        )
        assert service_name is not None, f"Failed to setup ECS service on cluster {ecs_cluster_name}"
        inference_result = request_mxnet_inference_squeezenet(public_ip_address)
        assert inference_result, f"Failed to perform inference at IP address: {public_ip_address}"
    finally:
        ecs_utils.tear_down_ecs_inference_service(cluster_arn, service_name, task_family, revision)
