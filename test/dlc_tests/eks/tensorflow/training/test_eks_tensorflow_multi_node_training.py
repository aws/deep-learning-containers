import os
import random
import re

from datetime import datetime

import pytest

from invoke.context import Context
from invoke import run
import test.test_utils.ec2 as ec2_utils
import test.test_utils.eks as eks_utils
from test.test_utils import is_pr_context, SKIP_PR_REASON, is_tf1


# Test only runs in region us-west-2, on instance type p3.16xlarge, on PR_EKS_CLUSTER_NAME_TEMPLATE cluster
# @pytest.mark.skipif(is_pr_context(), reason=SKIP_PR_REASON)
@pytest.mark.multinode("multinode(3)")
def test_eks_tensorflow_multi_node_training_gpu(tensorflow_training, example_only):
    # EKS multinode are failing on TF1 Pipeline due to scheduling issues.
    # TODO: Remove this line and add the required scheduling scheme.
    if is_tf1(tensorflow_training) :
        pytest.skip("Skipping it on TF1 currently as it is not able to do the pods scheduling properly")
    eks_cluster_size = "3"                                                        
    ec2_instance_type = "p3.16xlarge"

    eks_gpus_per_worker = ec2_utils.get_instance_num_gpus(instance_type=ec2_instance_type)

    _run_eks_tensorflow_multinode_training_resnet50_mpijob(tensorflow_training, eks_cluster_size, eks_gpus_per_worker)


def _run_eks_tensorflow_multinode_training_resnet50_mpijob(example_image_uri, cluster_size, eks_gpus_per_worker):
    """
    Run Tensorflow distributed training on EKS using horovod docker images with synthetic dataset
    :param example_image_uri:
    :param cluster_size:
    :param eks_gpus_per_worker:
    :return: None
    """
    user = Context().run("echo $USER").stdout.strip("\n")
    framework_version = re.search(r"\d+(\.\d+)+", example_image_uri).group()
    major_version = framework_version.split(".")[0]
    random.seed(f"{example_image_uri}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    unique_tag = f"{user}-{random.randint(1, 10000)}"
    namespace = f"tf{major_version}-multi-node-train-{'py2' if 'py2' in example_image_uri else 'py3'}-{unique_tag}"
    job_name = f"tf-resnet50-horovod-job-{unique_tag}"

    script_name = ("/deep-learning-models/models/resnet/tensorflow2/train_tf2_resnet.py" if major_version == "2" else
                   "/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py")

    args_to_pass = ('["--synthetic","--batch_size,128","--num_batches","100","--clear_log","2"]' if major_version == "2" else
                    '["--num_epochs=1","--synthetic"]')

    local_template_file_path = os.path.join(
        "eks",
        "eks_manifest_templates",
        "tensorflow",
        "training",
        "multi_node_gpu_training.yaml"
    )

    remote_yaml_file_path = os.path.join(os.sep, "tmp", f"tensorflow_multi_node_training_{unique_tag}.yaml")

    replace_dict = {
        "<JOB_NAME>": job_name,
        "<NUM_WORKERS>": cluster_size,
        "<CONTAINER_IMAGE>": example_image_uri,
        "<SCRIPT_NAME>": script_name,
        "<ARGS>": args_to_pass,
        "<GPUS>": str(eks_gpus_per_worker)
    }

    eks_utils.write_eks_yaml_file_from_template(local_template_file_path, remote_yaml_file_path, replace_dict)

    _run_eks_tensorflow_multi_node_training_mpijob(namespace, job_name, remote_yaml_file_path)


def _run_eks_tensorflow_multi_node_training_mpijob(namespace, job_name, remote_yaml_file_path):
    """
    Run Tensorflow distributed training on EKS using horovod docker images using MPIJob
    :param namespace:
    :param job_name:
    :param remote_yaml_file_path:
    :return: None
    """
    pod_name = None
    run(f"kubectl create namespace {namespace}")

    try:
        training_job_start = run(f"kubectl create -f {remote_yaml_file_path} -n {namespace}", warn=True)
        if training_job_start.return_code:
            raise RuntimeError(f"Failed to start {job_name}:\n{training_job_start.stderr}")

        eks_utils.LOGGER.info("Check pods")
        run(f"kubectl get pods -n {namespace} -o wide")

        complete_pod_name = eks_utils.is_mpijob_launcher_pod_ready(namespace, job_name)

        _, pod_name = complete_pod_name.split("/")
        eks_utils.LOGGER.info(f"The Pods have been created and the name of the launcher pod is {pod_name}")

        eks_utils.LOGGER.info(f"Wait for the {job_name} job to complete")
        if eks_utils.is_eks_multinode_training_complete(remote_yaml_file_path, namespace, pod_name, job_name):
            eks_utils.LOGGER.info(f"Wait for the {pod_name} pod to reach completion")
            distributed_out = run(f"kubectl logs -n {namespace} -f {complete_pod_name}").stdout
            eks_utils.LOGGER.info(distributed_out)
    finally:
        eks_utils.eks_multinode_cleanup(remote_yaml_file_path, namespace)