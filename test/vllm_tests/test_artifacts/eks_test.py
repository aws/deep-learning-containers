#!/usr/bin/env python3

import logging
import time
from invoke import run

logger = logging.getLogger(__name__)

class VllmEksTest:
    def __init__(self):
        pass

    def run_tests(self):
        try:
            self.deploy_vllm_service()
            self.test_vllm_api()
            return True
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False


    def deploy_vllm_service(self):
        logger.info("Deploying vLLM service...")

        # first, wait until the AWS Load Balancer Controller is running
        logger.info("Waiting for AWS Load Balancer Controller to be ready...")
        max_retries = 20  # 10 minutes total
        retry_count = 0
        
        while retry_count < max_retries:
            result = run("kubectl get pods -n kube-system | grep aws-load-balancer-controller", warn=True)
            if "aws-load-balancer-controller" in result.stdout:
                # count total and running ALB controller pods
                all_alb_pods = [
                    line for line in result.stdout.split("\n")
                    if "aws-load-balancer-controller" in line and line.strip()
                ]
                running_alb_pods = [
                    line for line in all_alb_pods
                    if "Running" in line
                ]
                if all_alb_pods and len(running_alb_pods) == len(all_alb_pods):
                    logger.info(f"All {len(running_alb_pods)} AWS Load Balancer Controller pods are running")
                    logger.info("AWS Load Balancer Controller is ready")
                    break
                else:
                    logger.info(f"ALB controller pods: {len(running_alb_pods)}/{len(all_alb_pods)} running")
            
            retry_count += 1
            logger.info(f"ALB controller not ready yet, waiting... (attempt {retry_count}/{max_retries})")
            time.sleep(30)
        
        if retry_count >= max_retries:
            raise Exception("AWS Load Balancer Controller pods failed to start after 10 minutes")

        # apply the LeaderWorkerSet
        run("cd aws-vllm-dlc-blog-repo && kubectl apply -f vllm-deepseek-32b-lws.yaml")
        # apply the ingress
        run("cd aws-vllm-dlc-blog-repo && kubectl apply -f vllm-deepseek-32b-lws-ingress.yaml")
        
        # monitor pod status until Running (can take 15-30 minutes for large GPU images + model loading)
        logger.info("Waiting for vLLM pods to reach Running status...")
        logger.info("This may take 15-30 minutes for container image pull and model loading")
        
        max_retries = 60  # 30 minutes total
        retry_count = 0
        
        while retry_count < max_retries:
            result = run("kubectl get pods -l app=vllm-deepseek-32b-lws", warn=True)
            if "vllm-deepseek-32b-lws" in result.stdout:
                # count total and running vLLM pods
                all_vllm_pods = [
                    line for line in result.stdout.split("\n")
                    if "vllm-deepseek-32b-lws" in line and line.strip() and "NAME" not in line
                ]
                running_vllm_pods = [
                    line for line in all_vllm_pods
                    if "Running" in line
                ]
                if all_vllm_pods and len(running_vllm_pods) == len(all_vllm_pods):
                    logger.info(f"All {len(running_vllm_pods)} vLLM pods are running")
                    logger.info("vLLM service is ready")
                    break
                else:
                    statuses = []
                    for line in all_vllm_pods:
                        parts = line.split()
                        if len(parts) >= 3:
                            pod_name = parts[0]
                            status = parts[2]
                            statuses.append(f"{pod_name}: {status}")
                    logger.info(f"vLLM pods status: {', '.join(statuses)}")
            
            retry_count += 1
            logger.info(f"vLLM pods not ready yet, waiting... (attempt {retry_count}/{max_retries})")
            time.sleep(30)
        
        if retry_count >= max_retries:
            raise Exception("vLLM pods failed to reach Running status after 30 minutes")

        logger.info("vLLM service deployed successfully")


    def test_vllm_api(self):
        logger.info("Testing vLLM API...")

        endpoint = run(
            "kubectl get ingress vllm-deepseek-32b-lws-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
        ).stdout.strip()
        logger.info(f"vLLM API endpoint: {endpoint}")

        # Test 1: completions API
        logger.info("Testing completions API...")
        result = run(
            f"""curl -X POST http://{endpoint}/v1/completions \
                -H "Content-Type: application/json" \
                -d '{{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "prompt": "Hello, how are you?", "max_tokens": 50, "temperature": 0.7}}'
            """
        )
        assert '"object":"text_completion"' in result.stdout, "vLLM completions API test failed"
        logger.info("Completions API test passed")
        
        # Test 2: chat completions API
        logger.info("Testing chat completions API...")
        result = run(
            f"""curl -X POST http://{endpoint}/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d '{{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "messages": [{{"role": "user", "content": "What are the benefits of using FSx Lustre with EKS?"}}], "max_tokens": 100, "temperature": 0.7}}'
            """
        )
        assert '"object":"chat.completion"' in result.stdout, "vLLM chat completions API test failed"
        logger.info("Chat completions API test passed")
        
        logger.info("All vLLM API tests passed successfully")
