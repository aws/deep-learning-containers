import logging, json, sys
import ruamel.yaml as yaml
from retrying import retry
from invoke import run
import base64

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

EKS_NVIDIA_PLUGIN_VERSION = "1.12"
# https://docs.aws.amazon.com/eks/latest/userguide/eks-optimized-ami.html
EKS_AMI_ID = {
    "cpu": "ami-0d3998d69ebe9b214",
    "gpu": "ami-0484012ada3522476"
}
SSH_PUBLIC_KEY_NAME = "dlc-ec2-keypair-prod"


def retry_if_value_error(exception):
    """Return True if we should retry (in this case when it's an ValueError), False otherwise"""
    return isinstance(exception, ValueError)


@retry(
    stop_max_attempt_number=40,
    wait_fixed=60000,
    retry_on_exception=retry_if_value_error,
)
def is_eks_training_complete(pod_name):
    """Function to check if the pod status has reached 'Completion'
    Args:
        pod_name: str
    """

    run_out = run("kubectl get pod {} -o json".format(pod_name))
    pod_info = json.loads(run_out.stdout)

    if "containerStatuses" in pod_info["status"]:
        container_status = pod_info["status"]["containerStatuses"][0]
        LOGGER.info("Container Status: %s", container_status)
        if container_status["name"] == pod_name:
            if "terminated" in container_status["state"]:
                if container_status["state"]["terminated"]["reason"] == "Completed":
                    LOGGER.info("SUCCESS: The container terminated.")
                    return True
                elif container_status["state"]["terminated"]["reason"] == "Error":
                    error_out = run("kubectl logs {}".format(pod_name)).stdout
                    # delete pod in case of error
                    run("kubectl delete pods {}".format(pod_name))
                    LOGGER.error(
                        "ERROR: The container run threw an error and terminated. "
                        "kubectl logs: %s",
                        error_out,
                    )
                    raise AttributeError("Container Error!")
            elif (
                "waiting" in container_status["state"]
                and container_status["state"]["waiting"]["reason"] == "CrashLoopBackOff"
            ):
                error_out = run("kubectl logs {}".format(pod_name)).stdout
                # delete pod in case of error
                run("kubectl delete pods {}".format(pod_name))
                LOGGER.error(
                    "ERROR: The container run threw an error in waiting state. "
                    "kubectl logs: %s",
                    error_out,
                )
                raise AttributeError("Error: CrashLoopBackOff!")
            elif (
                "waiting" in container_status["state"]
                or "running" in container_status["state"]
            ):
                LOGGER.info(
                    "IN-PROGRESS: Container is either Creating or Running. Waiting to complete..."
                )
                raise ValueError("IN-PROGRESS: Retry.")

    return False

def write_eks_yaml_file_from_template(local_template_file_path, remote_yaml_file_path, search_replace_dict):
    """Function that does a simple replace operation based on the search_replace_dict on the template file contents
    and writes the final yaml file to remote_yaml_path
    Args:
        local_template_path, remote_yaml_path: str
        search_replace_dict: dict
    """
    with open(local_template_file_path, "r") as yaml_file:
        yaml_data = yaml_file.read()

    for key, value in search_replace_dict.items():
        yaml_data = yaml_data.replace(key, value)
        print("***********")
        print(yaml_data)

    with open(remote_yaml_file_path, 'w') as yaml_file:
        yaml_file.write(yaml_data)

    LOGGER.info("Copied generated yaml file to %s", remote_yaml_file_path)

def is_eks_cluster_active(eks_cluster_name):
    """Function to verify if the default eks cluster is up and running.
    Args:
        eks_cluster_name: str
    Return:
        if_active: bool, true if status is active
    """
    if_active = False

    eksctl_check_cluster_command = """eksctl get cluster {} -o json
    """.format(eks_cluster_name)

    run_out = run(eksctl_check_cluster_command, warn_only=True)

    if run_out.return_code == 0:
        cluster_info = json.loads(run_out.stdout)[0]
        if_active = (cluster_info['Status'] == 'ACTIVE')

    return if_active

def eks_write_kubeconfig(eks_cluster_name, region="us-west-2"):
    """Function that writes the aws eks configuration for the specified cluster in the file ~/.kube/config
    This file is used by the kubectl and ks utilities along with aws-iam-authenticator to authenticate with aws
    and query the eks cluster.
    Note: This function assumes the cluster is 'ACTIVE'. Please use check_eks_cluster_status() to obtain status
    of the cluster.
    Args:
        eks_cluster_name, region: str
    """
    eksctl_write_kubeconfig_command = """eksctl utils write-kubeconfig \
                                         --name {} --region {}""".format(eks_cluster_name, region)
    run(eksctl_write_kubeconfig_command)

    run(f"aws eks --region us-west-2 update-kubeconfig --name {eks_cluster_name} --kubeconfig /root/.kube/config --role-arn arn:aws:iam::669063966089:role/nikhilsk-eks-test-role")

    LOGGER.info("kubeconfig successfully written to folder ~/.kube/config")
    run("cat /root/.kube/config", warn=True)

@retry(stop_max_attempt_number=4, wait_fixed=60000)
def create_eks_cluster_nodegroup(eks_cluster_name, processor_type, num_nodes,
                                 instance_type, ssh_public_key_name, region="us-west-2"):
    """Function to create and attach a nodegroup to an existing EKS cluster.
    Args:
        eks_cluster_name, processor_type, num_nodes, instance_type, ssh_public_key_name: str
    """

    eksctl_create_nodegroup_command = """eksctl create nodegroup \
                      --cluster {} \
                      --node-ami {} \
                      --nodes {} \
                      --node-type={} \
                      --timeout=40m \
                      --ssh-access \
                      --ssh-public-key {} \
                      --region {}""".format(eks_cluster_name, EKS_AMI_ID[processor_type], num_nodes, instance_type,
                                            ssh_public_key_name, region)

    run(eksctl_create_nodegroup_command)

    LOGGER.info("EKS cluster nodegroup created successfully, with the following parameters \
                cluster_name: {} \
                ami-id: {} \
                num_nodes: {} \
                instance_type: {} \
                ssh_public_key: {}".format(eks_cluster_name, EKS_AMI_ID[processor_type], num_nodes, instance_type, ssh_public_key_name))

# def delete_eks_cluster(eks_cluster_name):
#     """Function to delete the EKS cluster, if it exists. Additionally, the function cleans up any cloudformation stacks
#     that are dangling.
#     Args:
#         eks_cluster_name: str
#     """
#
#     run("eksctl delete cluster {} --wait".format(eks_cluster_name), warn_only=True)
#
#     cfn_stack_names = cloudformation_utils.list_cfn_stack_names(cfg.ami_materialset)
#     for stack_name in cfn_stack_names:
#         if eks_cluster_name in stack_name:
#             LOGGER.info("Deleting dangling cloudformation stack: {}".format(stack_name))
#             cloudformation_utils.delete_cfn_stack_and_wait(stack_name, cfg.ami_materialset)

@retry(stop_max_attempt_number=2, wait_fixed=60000)
def create_eks_cluster(eks_cluster_name, processor_type, num_nodes,
                       instance_type, ssh_public_key_name, region="us-west-2"):
    """Function to setup an EKS cluster using eksctl. The AWS credentials used to perform eks operations
    are that the user deepamiuser-beta as used in other functions. The 'deeplearning-ami-beta' public key
    will be used to access the nodes created as EC2 instances in the EKS cluster.
    Note: eksctl creates a cloudformation stack by the name of eksctl-${eks_cluster_name}-cluster.
    Args:
        eks_cluster_name, processor_type, num_nodes, instance_type, ssh_public_key_name: str
    """
    # delete_eks_cluster(eks_cluster_name)

    eksctl_create_cluster_command = """eksctl create cluster {} \
                                  --node-ami {} \
                                  --nodes {} \
                                  --node-type={} \
                                  --timeout=40m \
                                  --ssh-access \
                                  --ssh-public-key {} \
                                  --region {} \
                                  """.format(eks_cluster_name, EKS_AMI_ID[processor_type], num_nodes,
                                             instance_type, ssh_public_key_name, region)
    # In us-east-1 you are likely to get UnsupportedAvailabilityZoneException, if the allocated zones is us-east-1e as it does not support AmazonEKS
    if region == "us-east-1":
        eksctl_create_cluster_command += """--zones=us-east-1a,us-east-1b,us-east-1d \
                                         """
    eksctl_create_cluster_command += """--auto-kubeconfig"""
    run(eksctl_create_cluster_command)

    LOGGER.info("EKS cluster created successfully, with the following parameters\
                cluster_name: {} \
                ami-id: {} \
                num_nodes: {} \
                instance_type: {} \
                ssh_public_key: {}".format(eks_cluster_name, EKS_AMI_ID[processor_type], num_nodes, instance_type, ssh_public_key_name))

def setup_eks_cluster(processor_type, instance_type, num_nodes, eks_cluster_name):
    """Function to start, setup and verify the status of EKS cluster. It verifies if the cluster
    exists, if not, creates one. If a cluster exists and is 'ACTIVE', it verifies if the cluster
    nodegroup exists and attaches the nodegroup required. For the purpose of this test, the EKS
    cluster is associated with only one nodegroup.
    Args:
        processor_type, instance_type, num_nodes: str
    """


    if  not is_eks_cluster_active(eks_cluster_name):
        LOGGER.info("No associated nodegroup found for cluster: %s. Creating nodegroup.", eks_cluster_name)
    else:
        LOGGER.info("No active cluster named %s found. Creating the cluster.", eks_cluster_name)
        create_eks_cluster(eks_cluster_name, processor_type, num_nodes, instance_type, SSH_PUBLIC_KEY_NAME)

    eks_write_kubeconfig(eks_cluster_name)

    run("kubectl delete all --all", warn_only=True)

    if processor_type == "gpu":
        run("kubectl apply -f https://raw.githubusercontent.com/NVIDIA"
            "/k8s-device-plugin/v{}/nvidia-device-plugin.yml".format(EKS_NVIDIA_PLUGIN_VERSION))

    LOGGER.info("Cluster is active and associated nodegroup configured. "
                "Kubeconfig has been updated. EKS setup complete.")

def apply_aws_credentials_on_eks_pods(namespace):
    """Apply the credentials from MATERIALSET_FOR_EKS_USER. Can be parameterized later if different materialsets are used
    Args:
        namespace: str
    """
    local_secret_template_file_path = "container_tests/eks_manifest_templates/aws_access/secret.yaml"
    remote_secret_yaml_file_path = "/tmp/aws_access_secret.yaml"
    # aws_access_key_id, aws_secret_access_key = utils.get_private_key(MATERIALSET_FOR_EKS_USER)
    #
    # secret_search_replace_dict = {
    #     "<AWS_ACCESS_KEY_ID_BASE64>": base64.b64encode(aws_access_key_id.encode()),
    #     "<AWS_SECRET_ACCESS_KEY_BASE64>": base64.b64encode(aws_secret_access_key.encode())
    #     }
    #
    # write_eks_yaml_file_from_template(local_secret_template_file_path, remote_secret_yaml_file_path, secret_search_replace_dict)
    #
    # # Apply the secret
    # run("kubectl -n {} apply -f {}".format(namespace, remote_secret_yaml_file_path))

@retry(stop_max_attempt_number=5, wait_fixed=30000, retry_on_exception=retry_if_value_error)
def is_service_running(namespace, selector_name):
    """Check if the service pod is running
    Args:
        namespace, selector_name: str
    """
    run_out = run("kubectl get pods -n {} --selector=app={} -o jsonpath='{{.items[0].status.phase}}' ".format(namespace, selector_name), warn_only=True)

    if run_out.stdout == "Running":
        return True
    else:
        raise ValueError("Service not running yet, try again")

def run_inference_service_on_eks(docker_image_build_id, framework, processor, namespace, selector_name, eks_gpus_per_worker, inference_search_replace_dict, apply_credentials=False):
    """Run inference against the model specified
    Args:
        framework, processor, namespace, selector_name: str
        eks_gpus_per_worker: num
        inference_search_replace_dict: dict
        apply_credentials: boolean
    """

    run("kubectl delete namespace {}".format(namespace), warn_only=True)
    run("kubectl create namespace {}".format(namespace))

    local_inference_template_file_path = "container_tests/eks_manifest_templates/{}/" \
                                         "inference/single_node_{}_inference.yaml".format(framework, processor)
    remote_inference_yaml_file_path = "/tmp/{}_single_node_{}_inference_manifest.yaml".format(framework, processor)

    inference_search_replace_dict["<SELECTOR_NAME>"] = selector_name
    inference_search_replace_dict["<DOCKER_IMAGE_BUILD_ID>"] = docker_image_build_id

    # TODO: NUM_GPUS should be around: str(eks_gpus_per_worker). However, this causes 'insufficient nvidia.com/gpu' error.
    # Hardcoding this to "1". Tracking SIM: https://sim.amazon.com/issues/DLAMI-148
    if processor == "gpu":
        inference_search_replace_dict["<NUM_GPUS>"] = "1"

    if apply_credentials:
        apply_aws_credentials_on_eks_pods(namespace)

    write_eks_yaml_file_from_template(local_inference_template_file_path, remote_inference_yaml_file_path, inference_search_replace_dict)

    # Apply the inference job
    run("kubectl -n {} apply -f {}".format(namespace, remote_inference_yaml_file_path))

    try:
        is_service_running(namespace, selector_name)
    except ValueError as excp:
        LOGGER.error("Service is not running: %s", excp)

    LOGGER.info("EKS service is up and running. Ready for inference.")