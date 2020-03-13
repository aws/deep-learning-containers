import datetime

import test.test_utils.ecs as ecs_utils
import test.test_utils.ec2 as ec2_utils
from test.test_utils import request_mxnet_inference_squeezenet


def test_dummy(mxnet_inference):
    print(mxnet_inference)


def test_ecs_mxnet_inference(mxnet_inference, region):
    processor = 'gpu' if 'gpu' in mxnet_inference else 'cpu'
    framework = 'mxnet'
    job = 'inference'
    worker_instance_type = 'p3.8xlarge' if processor == 'gpu' else 'c5.18xlarge'
    cluster_arn = worker_instance_id = None
    try:
        datetime_suffix = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
        cluster_name = f'{framework}-{job}-{processor}-test-cluster-{datetime_suffix}'

        cluster_arn = ecs_utils.create_ecs_cluster(cluster_name, region=region)
        worker_instance_id, public_ip_address = ecs_utils.attach_ecs_worker_node(
            worker_instance_type, ecs_utils.ECS_AMI_ID[processor], cluster_name, cluster_arn, region=region)

        model_names = ["squeezenet"]
        num_cpus = ec2_utils.get_instance_num_cpus(worker_instance_id, region=region)
        num_gpus = (str(ec2_utils.get_instance_num_gpus(worker_instance_id, region=region)) if processor == 'gpu'
                    else None)
        # We assume that about 80% of RAM is free on the instance, since we are not directly querying it to find out
        # what the memory utilization is.
        memory = int(ec2_utils.get_instance_memory(worker_instance_id, region=region) * 0.8)
        squeezenet_test_args = [request_mxnet_inference_squeezenet, public_ip_address]

        tests_results = ecs_utils.ecs_inference_test_executor(mxnet_inference, framework, job, processor,
                                                              cluster_name, cluster_arn, datetime_suffix, model_names,
                                                              num_cpus*1024, memory, num_gpus, [squeezenet_test_args])

        assert all(tests_results), f'Tests failed - {tests_results}'
    finally:
        ecs_utils.cleanup_worker_node_cluster(worker_instance_id, cluster_arn)
