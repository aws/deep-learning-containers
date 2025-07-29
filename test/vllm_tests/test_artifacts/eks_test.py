import json
import logging
import os
import sys
import time
from invoke import run
import requests
import boto3
from botocore.exceptions import ClientError
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
VLLM_NAMESPACE = "vllm"
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
        eks_client = boto3.client("eks", region_name=AWS_REGION)
        vpc_id = eks_client.describe_cluster(name=CLUSTER_NAME)["cluster"]["resourcesVpcConfig"][
            "vpcId"
        ]
        alb_sg_name = f"{CLUSTER_NAME}-alb-sg"

        # Get ALB security group
        response = ec2_client.describe_security_groups(
            Filters=[
                {"Name": "group-name", "Values": [alb_sg_name]},
                {"Name": "vpc-id", "Values": [vpc_id]},
            ]
        )

        if not response["SecurityGroups"]:
            raise Exception(f"Security group {alb_sg_name} not found")

        alb_sg = response["SecurityGroups"][0]["GroupId"]
        user_ip = requests.get("https://checkip.amazonaws.com").text.strip()

        return alb_sg, user_ip

    except Exception as e:
        LOGGER.error(f"Failed to get ALB security group or current IP: {str(e)}")
        raise


def check_ip_rule_exists(security_group_rules, ip_address):
    """
    Check if an IP rule exists in security group rules
    """
    if not security_group_rules:
        return False

    for rule in security_group_rules:
        if (
            rule.get("FromPort") == 80
            and rule.get("ToPort") == 80
            and rule.get("IpProtocol") == "tcp"
            and "IpRanges" in rule
        ):
            for ip_range in rule.get("IpRanges", []):
                if ip_range.get("CidrIp") == f"{ip_address}/32":
                    LOGGER.info(f"Found existing rule for IP {ip_address}")
                    return True
    return False


def authorize_ingress(ec2_client, group_id, ip_address):
    try:
        response = ec2_client.describe_security_groups(GroupIds=[group_id])
        if response.get("SecurityGroups") and response["SecurityGroups"]:
            existing_rules = response["SecurityGroups"][0].get("IpPermissions", [])
            if check_ip_rule_exists(existing_rules, ip_address):
                LOGGER.info("Ingress rule already exists, skipping creation.")
                return

        ec2_client.authorize_security_group_ingress(
            GroupId=group_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 80,
                    "ToPort": 80,
                    "IpRanges": [
                        {
                            "CidrIp": f"{ip_address}/32",
                            "Description": "Temporary access for vLLM testing",
                        }
                    ],
                }
            ],
        )
        LOGGER.info("Ingress rule added successfully.")
    except ClientError as e:
        LOGGER.error(f"Failed to authorize ingress: {str(e)}")
        raise


@retry(
    stop_max_attempt_number=40,
    wait_fixed=300000,
    retry_on_exception=retry_if_value_error,
)
def wait_for_pods_ready():
    run_out = run(
        f"kubectl get pods -l leaderworkerset.sigs.k8s.io/name=vllm-deepseek-32b-lws -n {VLLM_NAMESPACE} -o json"
    )
    pods = json.loads(run_out.stdout)

    if not pods.get("items"):
        LOGGER.info("No pods found yet...")
        raise ValueError("Pods not created yet")

    for pod in pods["items"]:
        pod_name = pod["metadata"]["name"]
        pod_phase = pod["status"]["phase"]
        LOGGER.info(f"Checking pod {pod_name}: {pod_phase}")
        if pod_phase != "Running":
            if pod.get("status", {}).get("containerStatuses"):
                container_status = pod["status"]["containerStatuses"][0]
                if (
                    container_status.get("state", {}).get("waiting", {}).get("reason")
                    == "CrashLoopBackOff"
                ):
                    error_out = run(f"kubectl logs {pod_name} -n {VLLM_NAMESPACE}").stdout
                    LOGGER.error(f"Pod {pod_name} crashed: {error_out}")
                    raise AttributeError(f"Container Error in pod {pod_name}")
            raise ValueError(f"Pod {pod_name} not ready yet")

        # Check if container is ready 1/1
        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        if not container_statuses or not container_statuses[0].get("ready", False):
            LOGGER.info(f"Pod {pod_name} is running but container not ready")
            raise ValueError(f"Container in pod {pod_name} not ready yet")

    return True


@retry(
    stop_max_attempt_number=20,
    wait_fixed=30000,
    retry_on_exception=retry_if_value_error,
)
def wait_for_ingress_ready(name, namespace=VLLM_NAMESPACE):
    run_out = run(f"kubectl get ingress {name} -n {namespace} -o json")
    ingress = json.loads(run_out.stdout)

    if not ingress.get("status", {}).get("loadBalancer", {}).get("ingress"):
        LOGGER.info("Waiting for ALB to be ready...")
        raise ValueError("Ingress ALB not ready yet")
    return ingress["status"]["loadBalancer"]["ingress"][0]["hostname"]


def test_api_endpoint(endpoint, api_type, max_retries=5, wait_time=60):
    for attempt in range(max_retries):
        try:
            LOGGER.info(f"Attempt {attempt + 1} of {max_retries} to test endpoint")
            if api_type not in ["completions", "chat_completions"]:
                raise ValueError(f"Invalid API type: {api_type}")
            payload = {
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "max_tokens": 100,
                "temperature": 0.7,
            }

            if api_type == "completions":
                payload["prompt"] = "Hello, how are you?"
                url = f"http://{endpoint}/v1/completions"
            elif api_type == "chat_completions":
                payload["messages"] = [
                    {
                        "role": "user",
                        "content": "What are the benefits of using FSx Lustre with EKS?",
                    }
                ]
                url = f"http://{endpoint}/v1/chat/completions"

            LOGGER.info(f"Sending request to {url} with payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            response_json = response.json()
            LOGGER.info(f"Received response: {json.dumps(response_json, indent=2)}")
            return response.json()
        except requests.RequestException as e:
            LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                LOGGER.info(f"Waiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)
            else:
                LOGGER.error("All attempts failed")
                # Add debugging info before failing
                LOGGER.info("Getting debug information...")
                try:
                    run("kubectl get pods -n vllm")
                    run("kubectl describe pods -n vllm")
                    run("kubectl logs -n vllm -l role=leader")
                    run("kubectl get svc -n vllm")
                    run("kubectl describe ingress -n vllm")
                except Exception as debug_e:
                    LOGGER.error(f"Error getting debug info: {str(debug_e)}")
                raise


def validate_api_response(result):
    required_fields = ["id", "object", "choices", "usage"]
    return all(field in result for field in required_fields)


@retry(
    stop_max_attempt_number=30,
    wait_fixed=300000,
    retry_on_exception=retry_if_value_error,
)
def wait_for_scale_down():
    run_out = run("kubectl get nodes -l role=large-model-worker -o json")
    nodes = json.loads(run_out.stdout)
    if nodes.get("items"):
        LOGGER.info(f"Still have {len(nodes['items'])} worker nodes, waiting...")
        raise ValueError("Nodes still present")
    return True


def cleanup(ec2_client, alb_sg, user_ip):
    try:
        LOGGER.info("Revoking ingress rule...")
        try:
            response = ec2_client.describe_security_groups(GroupIds=[alb_sg])
            if response.get("SecurityGroups") and response["SecurityGroups"]:
                existing_rules = response["SecurityGroups"][0].get("IpPermissions", [])
                if check_ip_rule_exists(existing_rules, user_ip):
                    ec2_client.revoke_security_group_ingress(
                        GroupId=alb_sg,
                        IpPermissions=[
                            {
                                "IpProtocol": "tcp",
                                "FromPort": 80,
                                "ToPort": 80,
                                "IpRanges": [{"CidrIp": f"{user_ip}/32"}],
                            }
                        ],
                    )
                    LOGGER.info("Ingress rule revoked successfully")
                else:
                    LOGGER.info("No matching ingress rule found, skipping revocation")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == "InvalidPermission.NotFound":
                LOGGER.warning("Ingress rule not found, skipping revoke operation")
            else:
                LOGGER.error(f"Failed to revoke ingress rule: {str(e)}")

        LOGGER.info("Deleting kubernetes resources...")
        try:
            run(f"kubectl delete -f {LWS_INGRESS_YAML} -n {VLLM_NAMESPACE}")
            run(f"kubectl delete -f {LWS_YAML} -n {VLLM_NAMESPACE}")
        except Exception as e:
            LOGGER.warning(f"Resource deletion warning: {str(e)}")

        LOGGER.info("Waiting for nodes to scale down...")
        wait_for_scale_down()

    except Exception as e:
        LOGGER.error(f"Cleanup failed: {str(e)}")
        raise


def test_vllm_on_eks(image):
    """
    Run vLLM tests on EKS using specified image
    """
    try:
        LOGGER.info(f"Starting EKS tests with image: {image}")

        # Verify EKS setup and make sure cluster is active
        eks_setup()
        if not is_eks_cluster_active(CLUSTER_NAME):
            raise Exception(f"EKS cluster {CLUSTER_NAME} is not active")

        eks_write_kubeconfig(CLUSTER_NAME, AWS_REGION)
        ec2_client = get_ec2_client(AWS_REGION)

        LOGGER.info("Getting security group and IP info...")
        alb_sg, user_ip = get_sg_and_ip_info()

        # Deploy vLLM using kubectl apply
        LOGGER.info("Deploying vLLM...")
        run(f"sed -i 's|<image>|{image}|g' {LWS_YAML}")
        run(f"kubectl apply -f {LWS_YAML} -n {VLLM_NAMESPACE}")
        run(f"sed -i 's|{image}|<image>|g' {LWS_YAML}")

        # Update and deploy ingress
        LOGGER.info("Deploying ingress...")
        run(f"sed -i 's|<sg-id>|{alb_sg}|g' {LWS_INGRESS_YAML}")
        run(f"kubectl apply -f {LWS_INGRESS_YAML} -n {VLLM_NAMESPACE}")
        run(f"sed -i 's|{alb_sg}|<sg-id>|g' {LWS_INGRESS_YAML}")

        LOGGER.info("Adding ingress rule...")
        authorize_ingress(ec2_client, alb_sg, user_ip)

        LOGGER.info("Waiting for pods to be ready...")
        wait_for_pods_ready()

        LOGGER.info("Waiting for ALB endpoint...")
        endpoint = wait_for_ingress_ready("vllm-deepseek-32b-lws-ingress")
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
        LOGGER.info("Waiting 5 minutes before cleanup to allow for debugging...")
        time.sleep(300)
        raise

    finally:
        time.sleep(60)  # 1 minute buffer before starting cleanup
        LOGGER.info("Cleaning up...")
        cleanup(ec2_client, alb_sg, user_ip)
