import datetime
import pytest

import test.test_utils.ecs as ecs_utils
import test.test_utils.ec2 as ec2_utils
from test.test_utils import request_mxnet_inference_squeezenet
from test.test_utils import ECS_AML2_CPU_USWEST2, ECS_AML2_GPU_USWEST2


def test_gpu_dummy(mxnet_inference, gpu_only):
    print(mxnet_inference)


def test_cpu_dummy(mxnet_inference, cpu_only):
    print(mxnet_inference)


def test_ecs_mxnet_inference(mxnet_inference, ecs_cluster_name, region):
    processor = "gpu" if "gpu" in mxnet_inference else "cpu"
    framework = "mxnet"
    job = "inference"
    python_version = "py2" if "py2" in mxnet_inference else "py3"
    worker_ami_id = ECS_AML2_GPU_USWEST2 if processor == 'gpu' else ECS_AML2_CPU_USWEST2
    worker_instance_type = 'p3.8xlarge' if processor == 'gpu' else 'c5.18xlarge'
    cluster_arn = worker_instance_id = None
    service_name = task_family = revision = None
    try:
        datetime_suffix = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        ecs_cluster_name = f'mxnet-inference-{processor}-{python_version}-test-cluster-{datetime_suffix}'

        cluster_arn = ecs_utils.create_ecs_cluster(ecs_cluster_name, region=region)
        worker_instance_id, public_ip_address = ecs_utils.attach_ecs_worker_node(
            worker_instance_type, worker_ami_id, ecs_cluster_name, cluster_arn, region=region
        )

        model_names = ["squeezenet"]
        num_cpus = ec2_utils.get_instance_num_cpus(worker_instance_id, region=region)
        num_gpus = (str(ec2_utils.get_instance_num_gpus(worker_instance_id, region=region)) if processor == "gpu"
                    else None)
        # We assume that about 80% of RAM is free on the instance, since we are not directly querying it to find out
        # what the memory utilization is.
        memory = int(ec2_utils.get_instance_memory(worker_instance_id, region=region) * 0.8)

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
        ecs_utils.cleanup_worker_node_cluster(worker_instance_id, cluster_arn)
