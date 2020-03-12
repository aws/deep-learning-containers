import datetime

import test.dlc_tests.test_utils.ecs as ecs_utils
import test.dlc_tests.test_utils.ec2 as ec2_utils
from test.dlc_tests.test_utils.general import test_mxnet_inference_squeezenet


def test_dummy(mxnet_inference):
    print(mxnet_inference)


def test_ecs_mxnet_inference(mxnet_inference, framework, job, processor, region):
    worker_instance_type = 'p3.8xlarge' if processor == 'gpu' else 'c5.18xlarge'
    cluster_arn = worker_instance_id = None
    try:
        cluster_name = 'mxnet-inference-test-cluster'
        datetime_suffix = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')

        cluster_arn = ecs_utils.create_ecs_cluster(cluster_name, region=region)
        worker_instance_id, public_ip_address = ecs_utils.attach_ecs_worker_node(
            worker_instance_type, ecs_utils.ECS_AMI_ID[processor], cluster_name, cluster_arn, region=region)

        model_names = ["squeezenet"]
        num_cpus = ec2_utils.get_instance_num_cpus(worker_instance_id, region=region)
        num_gpus = ec2_utils.get_instance_num_gpus(worker_instance_id, region=region) if processor == 'gpu' else None
        memory = ec2_utils.get_instance_memory(worker_instance_id, region=region)
        squeezenet_test_args = [test_mxnet_inference_squeezenet, public_ip_address]

        tests_results = ecs_utils.ecs_inference_test_executor(mxnet_inference, framework, job, processor,
                                                              cluster_name, cluster_arn, datetime_suffix, model_names,
                                                              num_cpus*1024, memory, num_gpus, [squeezenet_test_args])

        assert all(tests_results), f'Tests failed - {tests_results}'
    finally:
        ecs_utils.cleanup_worker_node_cluster(worker_instance_id, cluster_arn)
