import os

from invoke.context import Context

from src.github import GitHubHandler
import test.test_utils.ec2 as ec2_utils
import test.test_utils.eks as eks_utils


# Test only runs in region us-west-2, on instance type p3.16xlarge, on PR_EKS_CLUSTER_NAME_TEMPLATE cluster
def test_eks_tensorflow_multi_node_training_gpu(tensorflow_training, py3_only, gpu_only, example):
    eks_cluster_size = 3
    ec2_instance_type = "p3.16xlarge"
    cluster_name = eks_utils.PR_EKS_CLUSTER_NAME_TEMPLATE.format("tensorflow")

    assert eks_utils.is_eks_cluster_active(cluster_name), f"EKS Cluster {cluster_name} is inactive. Exiting test"

    eks_gpus_per_worker = ec2_utils.get_instance_num_gpus(instance_type=ec2_instance_type)

    run_eks_tensorflow_multinode_training_resnet50_mpijob(tensorflow_training, eks_cluster_size, eks_gpus_per_worker)


def run_eks_tensorflow_multinode_training_resnet50_mpijob(example_image_uri, cluster_size, eks_gpus_per_worker):
    """
    Run Tensorflow distributed training on EKS using horovod docker images with synthetic dataset
    :param example_image_uri:
    :param cluster_size:
    :param eks_gpus_per_worker:
    :return: None
    """
    # Use the image tag as namespace to make it unique within the CodeBuild job
    unique_tag = example_image_uri.split(':')[-1].replace(".", "-")
    # namespace = f"tensorflow-multi-node-training-{unique_tag}"
    namespace = "default"
    job_name = f"tf-resnet50-horovod-job-{unique_tag}"
    command_to_run = ("mpirun,-mca,btl_tcp_if_exclude,lo,-mca,pml,ob1,-mca,btl,^openib,--bind-to,none,-map-by,slot,"
                      "-x,LD_LIBRARY_PATH,-x,PATH,-x,NCCL_SOCKET_IFNAME=eth0,-x,NCCL_DEBUG=INFO,python,"
                      "/deep-learning-models/models/resnet/tensorflow/train_imagenet_resnet_hvd.py")
    args_to_pass = "-- --num_epochs=1,--synthetic"
    path_to_ksonnet_app = f"~/tensorflow_multi_node_eks_test-{unique_tag}/"
    app_name = f"kubeflow-tf-hvd-mpijob-{unique_tag}"

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
    KUBEFLOW_VERSION = "v0.5.1"
    pod_name = None
    ctx = Context()
    github_handler = GitHubHandler("aws", "kubeflow")

    if not os.path.exists(path_to_ksonnet_app):
        ctx.run(f"mkdir -p {path_to_ksonnet_app}")

    with ctx.cd(path_to_ksonnet_app):
        ctx.run(f"rm -rf {app_name}")
        ctx.run(f"ks init {app_name}")

        with ctx.cd(app_name):
            # Check if the kubeflow registry exists and create. Registry will be available in each pod.
            registry_not_exist = ctx.run("ks registry list | grep kubeflow", warn=True)

            if registry_not_exist.return_code:
                github_token = github_handler.get_auth_token()
                os.environ["GITHUB_TOKEN"] = github_token
                ctx.run(f"ks registry add kubeflow github.com/kubeflow/kubeflow/tree/{KUBEFLOW_VERSION}/kubeflow")
                ctx.run(f"ks pkg install kubeflow/common@{KUBEFLOW_VERSION}")
                ctx.run(f"ks pkg install kubeflow/mpi-job@{KUBEFLOW_VERSION}")

            ctx.run("ks generate mpi-operator mpi-operator")
            # The latest mpi-operator docker image does not accept the gpus-per-node parameter
            # which is specified by the older spec file from v0.5.1.
            ctx.run("ks param set mpi-operator image mpioperator/mpi-operator:0.2.0")
            ctx.run("ks apply default -c mpi-operator")
            eks_utils.LOGGER.info(
                "The mpi-operator package must be applied to default before we can use mpiJob. "
                "Check status before moving on."
            )
            ctx.run("kubectl get crd")

            # Use Ksonnet to generate manifest files which are then applied to the default context.
            ctx.run(f"ks generate mpi-job-custom {job_name}")
            ctx.run(f"ks param set {job_name} replicas {cluster_size}")
            ctx.run(f"ks param set {job_name} gpusPerReplica {eks_gpus_per_worker}")
            ctx.run(f"ks param set {job_name} image {custom_image}")
            ctx.run(f"ks param set {job_name} command {command_to_run}")
            ctx.run(f"ks param set {job_name} args {args_to_pass}")

            try:
                # use `$ks show default` to see details.
                ctx.run("kubectl get pods -o wide")
                eks_utils.LOGGER.info("Apply the generated manifest to the default env.")
                ctx.run(f"ks apply default -c {job_name}")

                eks_utils.LOGGER.info("Check pods")
                ctx.run("kubectl get pods -o wide")

                eks_utils.LOGGER.info(
                    "First the mpi-operator and the n-worker pods will be created and then "
                    "the launcher pod is created in the end. Use retries until launcher "
                    "pod's name is available to read logs."
                )
                complete_pod_name = eks_utils.is_mpijob_launcher_pod_ready(ctx, job_name)

                _, pod_name = complete_pod_name.split("/")
                eks_utils.LOGGER.info(f"The Pods have been created and the name of the launcher pod is {pod_name}")

                eks_utils.LOGGER.info(f"Wait for the {job_name} job to complete")
                if eks_utils.is_eks_multinode_training_complete(ctx, namespace, pod_name, job_name):
                    eks_utils.LOGGER.info(f"Wait for the {pod_name} pod to reach completion")
                    distributed_out = ctx.run(f"kubectl logs -f {complete_pod_name}").stdout
                    eks_utils.LOGGER.info(distributed_out)
            finally:
                eks_utils.eks_multinode_cleanup(ctx, pod_name, job_name, namespace)
