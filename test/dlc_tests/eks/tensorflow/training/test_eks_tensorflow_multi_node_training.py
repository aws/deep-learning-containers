import os
import random

from invoke import run
import pytest

from test import test_utils
import test.test_utils.eks as eks_utils


@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
def test_eks_tensorflow_multi_node_training_gpu(tensorflow_training, ec2_instance_type, ec2_client, gpu_only, example):
    cluster_name = eks_utils.PR_EKS_CLUSTER_NAME_TEMPLATE.format("tensorflow")
    image_tag = tensorflow_training.split(":")[-1]
    py_version = "py2" if "py2" in image_tag else "py3"
    ec2_key_name = (f"test_eks_tensorflow_multi_node_training-{image_tag}-"
                    f"{os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')}-cpu-{py_version}")

    assert eks_utils.is_eks_cluster_active(cluster_name), f"EKS Cluster {cluster_name} is inactive. Exiting test"

    key_filename = test_utils.generate_ssh_keypair(ec2_client, ec2_key_name)

    run_eks_tensorflow_multinode_training_resnet50_mpijob(tensorflow_training, 3, 72)


def run_eks_tensorflow_multinode_training_resnet50_mpijob(example_image_uri, cluster_size, eks_gpus_per_worker):
    """
    Run Tensorflow distributed training on EKS using horovod docker images with synthetic dataset
    :param cluster_size:
    :param eks_gpus_per_worker:
    :return: None
    """
    namespace = "default"
    job_name = "tf-resnet50-horovod-job"
    command_to_run = ("mpirun,-mca,btl_tcp_if_exclude,lo,-mca,pml,ob1,-mca,btl,^openib,--bind-to,none,-map-by,slot,"
                      "-x,LD_LIBRARY_PATH,-x,PATH,-x,NCCL_SOCKET_IFNAME=eth0,-x,NCCL_DEBUG=INFO,python,"
                      "/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py")
    args_to_pass = "-- --num_epochs=1,--synthetic"
    path_to_ksonnet_app = "~/src/container_tests/"
    app_name = "kubeflow-tf-hvd-mpijob"

    run_eks_tensorflow_multi_node_training_mpijob(namespace, app_name, example_image_uri, job_name,
                                                  command_to_run, args_to_pass, path_to_ksonnet_app,
                                                  cluster_size, eks_gpus_per_worker)


def run_eks_tensorflow_multi_node_training_mpijob(namespace, app_name, custom_image, job_name,
                                                  command_to_run, args_to_pass, path_to_ksonnet_app,
                                                  cluster_size, eks_gpus_per_worker):
    """
    Run Tensorflow distributed training on EKS using horovod docker images using MPIJob
    :param namespace:
    :param app_name:
    :param custom_image:
    :param job_name:
    :param command_to_run:
    :param args_to_pass:
    :param path_to_ksonnet_app:
    :param cluster_size:
    :param eks_gpus_per_worker:
    :return: None
    """
    pass
