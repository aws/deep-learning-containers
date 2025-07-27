import json
import logging
import os
import sys
from invoke import run
import requests
import boto3
from botocore.exceptions import ClientError
from kubernetes import client, config
from retrying import retry
from test.test_utils.ec2 import get_ec2_client
from test.test_utils.eks import (
    retry_if_value_error,
    eks_setup,
    is_eks_cluster_active,
    eks_write_kubeconfig,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

AWS_REGION = "us-west-2"
CLUSTER_NAME = "dlc-vllm-PR"
TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LWS_YAML = os.path.join(TEST_DIR, "test_artifacts", "vllm-deepseek-32b-lws.yaml")
LWS_INGRESS_YAML = os.path.join(TEST_DIR, "test_artifacts", "vllm-deepseek-32b-lws-ingress.yaml")

def get_sg_and_ip_info():
    """
    Get ALB security group ID and current IP address
    """
    try:
        ec2_client = get_ec2_client(AWS_REGION)
        
        # Get VPC ID from EKS cluster
        eks_client = boto3.client('eks', region_name=AWS_REGION)
        vpc_id = eks_client.describe_cluster(
            name=CLUSTER_NAME
        )['cluster']['resourcesVpcConfig']['vpcId']
        alb_sg_name=f"{CLUSTER_NAME}-alb-sg"
        
        # Get ALB security group
        response = ec2_client.describe_security_groups(
            Filters=[
                {
                    'Name': 'group-name',
                    'Values': [alb_sg_name]
                },
                {
                    'Name': 'vpc-id',
                    'Values': [vpc_id]
                }
            ]
        )
        
        if not response['SecurityGroups']:
            raise Exception(f"Security group {alb_sg_name} not found")
            
        alb_sg = response['SecurityGroups'][0]['GroupId']
        user_ip = requests.get('https://checkip.amazonaws.com').text.strip()
        
        return alb_sg, user_ip
        
    except Exception as e:
        LOGGER.error(f"Failed to get ALB security group or current IP: {str(e)}")
        raise


def authorize_ingress(ec2_client, group_id, ip_address):
    """
    Authorize security group ingress using EC2 client
    """
    try:
        ec2_client.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [
                        {
                            'CidrIp': f'{ip_address}/32',
                            'Description': 'Temporary access for vLLM testing'
                        }
                    ]
                }
            ]
        )
    except ClientError as e:
        LOGGER.error(f"Failed to authorize ingress: {str(e)}")
        raise


@retry(
    stop_max_attempt_number=60,
    wait_fixed=300000,
    retry_on_exception=retry_if_value_error,
)
def wait_for_pods_ready(k8s_client):
    pods = k8s_client.list_namespaced_pod(
        namespace="default",
        label_selector="leaderworkerset.sigs.k8s.io/name=vllm-deepseek-32b-lws"
    )
    
    if not pods.items:
        LOGGER.info("No pods found yet...")
        raise ValueError("Pods not created yet")
    
    for pod in pods.items:
        LOGGER.info(f"Checking pod {pod.metadata.name}: {pod.status.phase}")
        if pod.status.phase != "Running":
            if pod.status.container_statuses:
                container_status = pod.status.container_statuses[0]
                if (container_status.state.waiting and 
                    container_status.state.waiting.reason == "CrashLoopBackOff"):
                    error_out = k8s_client.read_namespaced_pod_log(
                        pod.metadata.name, "default"
                    )
                    LOGGER.error(f"Pod {pod.metadata.name} crashed: {error_out}")
                    raise AttributeError(f"Container Error in pod {pod.metadata.name}")
            raise ValueError(f"Pod {pod.metadata.name} not ready yet")
    
    return True


@retry(
    stop_max_attempt_number=20,
    wait_fixed=30000,
    retry_on_exception=retry_if_value_error,
)
def wait_for_ingress_ready(k8s_networking_client, name, namespace = "default"):
    ingress = k8s_networking_client.read_namespaced_ingress(
        name=name,
        namespace=namespace
    )
    if not ingress.status.load_balancer.ingress:
        LOGGER.info("Waiting for ALB to be ready...")
        raise ValueError("Ingress ALB not ready yet")
    return ingress.status.load_balancer.ingress[0].hostname


def test_api_endpoint(endpoint, api_type):
    if api_type not in ["completions", "chat_completions"]:
        raise ValueError(f"Invalid API type: {api_type}")
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    if api_type == "completions":
        payload["prompt"] = "Hello, how are you?"
    elif api_type == "chat_completions":
        payload["messages"] = [{"role": "user", "content": "What are the benefits of using FSx Lustre with EKS?"}]
    
    response = requests.post(
        f"http://{endpoint}/v1/{api_type}",
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def validate_api_response(result) :
    required_fields = ['id', 'object', 'choices', 'usage']
    return all(field in result for field in required_fields)

@retry(
    stop_max_attempt_number=30,
    wait_fixed=300000,
    retry_on_exception=retry_if_value_error,
)
def wait_for_scale_down(k8s_client):
    nodes = k8s_client.list_node(
        label_selector="role=large-model-worker"
    )
    if nodes.items:
        LOGGER.info(f"Still have {len(nodes.items)} worker nodes, waiting...")
        raise ValueError("Nodes still present")
    return True


def cleanup(ec2_client, alb_sg, user_ip, k8s_client):
    try:
        LOGGER.info("Revoking ingress rule...")
        ec2_client.revoke_security_group_ingress(
            GroupId=alb_sg,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 80,
                    'ToPort': 80,
                    'IpRanges': [{'CidrIp': f'{user_ip}/32'}]
                }
            ]
        )
        
        LOGGER.info("Deleting kubernetes resources...")
        try:
            run(f"kubectl delete -f {LWS_INGRESS_YAML}", warn=True)
            run(f"kubectl delete -f {LWS_YAML}", warn=True)
        except Exception as e:
            LOGGER.warning(f"Resource deletion warning: {str(e)}")
        
        LOGGER.info("Waiting for nodes to scale down...")
        wait_for_scale_down(k8s_client)
        
    except Exception as e:
        LOGGER.error(f"Cleanup failed: {str(e)}")
        raise


def test_vllm_on_eks():
    """
    Run vLLM tests on EKS using image from YAML
    """
    try:
        LOGGER.info("Starting EKS tests with predefined image")

        # Verify EKS setup and make sure cluster is active
        eks_setup()
        if not is_eks_cluster_active(CLUSTER_NAME):
            raise Exception(f"EKS cluster {CLUSTER_NAME} is not active")
        
        eks_write_kubeconfig(CLUSTER_NAME, AWS_REGION)
        ec2_client = get_ec2_client(AWS_REGION)
        
        # Setup kubernetes client
        config.load_kube_config()
        k8s_client = client.CoreV1Api()
        k8s_networking_client = client.NetworkingV1Api()
        
        LOGGER.info("Getting security group and IP info...")
        alb_sg, user_ip = get_sg_and_ip_info()
        
        # Deploy vLLM using kubectl apply
        LOGGER.info("Deploying vLLM...")
        run(f"kubectl apply -f {LWS_YAML}", check=True)
        
        # Update and deploy ingress
        LOGGER.info("Deploying ingress...")
        run(f"sed -i 's|<sg-id>|{alb_sg}|g' {LWS_INGRESS_YAML}", check=True)
        run(f"kubectl apply -f {LWS_INGRESS_YAML}", check=True)
        run(f"sed -i 's|{alb_sg}|<sg-id>|g' {LWS_INGRESS_YAML}", check=True)
        
        LOGGER.info("Adding ingress rule...")
        authorize_ingress(ec2_client, alb_sg, user_ip)
        
        LOGGER.info("Waiting for pods to be ready...")
        wait_for_pods_ready(k8s_client)
        
        LOGGER.info("Waiting for ALB endpoint...")
        endpoint = wait_for_ingress_ready(
            k8s_networking_client, 
            "vllm-deepseek-32b-lws-ingress"
        )
        LOGGER.info(f"ALB endpoint: {endpoint}")
        
        # Run tests
        LOGGER.info("Testing completions API...")
        completions_result = test_api_endpoint(endpoint, "completions")
        if not validate_api_response(completions_result):
            raise ValueError("Completions test failed")
        
        LOGGER.info("Testing chat completions API...")
        chat_result = test_api_endpoint(endpoint, "chat_completions")
        if not validate_api_response(chat_result):
            raise ValueError("Chat completions test failed")
        
        LOGGER.info("All tests passed!")
        return True
        
    except Exception as e:
        LOGGER.error(f"Test failed: {str(e)}")
        raise
        
    finally:
        LOGGER.info("Cleaning up...")
        cleanup(ec2_client, alb_sg, user_ip, k8s_client)