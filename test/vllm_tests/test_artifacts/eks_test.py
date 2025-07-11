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
            logger.info("Starting vLLM EKS integration tests...")
            self.deploy_vllm_service()
            self.test_vllm_api()
            logger.info("All vLLM EKS tests completed successfully")
            return True
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return False


    def deploy_vllm_service(self):
        logger.info("Deploying vLLM service...")

        self._wait_for_load_balancer_controller()
        
        logger.info("Applying vLLM LeaderWorkerSet configuration...")
        run("kubectl apply -f test/vllm_tests/test_artifacts/vllm-deepseek-32b-lws.yaml")
        
        logger.info("Applying vLLM ingress configuration...")
        run("kubectl apply -f test/vllm_tests/test_artifacts/vllm-deepseek-32b-lws-ingress.yaml")
        
        self._wait_for_vllm_pods()
        
        logger.info("vLLM service deployed successfully")
    

    def _wait_for_load_balancer_controller(self):
        logger.info("Waiting for AWS Load Balancer Controller to be ready...")
        max_retries = 20  # 10 minutes total
        retry_count = 0
        
        while retry_count < max_retries:
            result = run("kubectl get pods -n kube-system | grep aws-load-balancer-controller", warn=True)
            if "aws-load-balancer-controller" in result.stdout:
                all_alb_pods = [
                    line for line in result.stdout.split("\n")
                    if "aws-load-balancer-controller" in line and line.strip()
                ]
                running_alb_pods = [
                    line for line in all_alb_pods if "Running" in line
                ]
                if all_alb_pods and len(running_alb_pods) == len(all_alb_pods):
                    logger.info(f"All {len(running_alb_pods)} AWS Load Balancer Controller pods are running")
                    return
                else:
                    logger.info(f"ALB controller pods: {len(running_alb_pods)}/{len(all_alb_pods)} running")
            
            retry_count += 1
            logger.info(f"ALB controller not ready yet, waiting... (attempt {retry_count}/{max_retries})")
            time.sleep(30)
        
        raise Exception("AWS Load Balancer Controller pods failed to start after 10 minutes")
    

    def _wait_for_vllm_pods(self):
        logger.info("Waiting for vLLM pods to reach Running status...")
        logger.info("This may take 15-30 minutes for container image pull and model loading")
        
        max_retries = 60  # 30 minutes total
        retry_count = 0
        
        while retry_count < max_retries:
            result = run("kubectl get pods -l app=vllm-deepseek-32b-lws", warn=True)
            if "vllm-deepseek-32b-lws" in result.stdout:
                all_vllm_pods = [
                    line for line in result.stdout.split("\n")
                    if "vllm-deepseek-32b-lws" in line and line.strip() and "NAME" not in line
                ]
                running_vllm_pods = [
                    line for line in all_vllm_pods if "Running" in line
                ]
                if all_vllm_pods and len(running_vllm_pods) == len(all_vllm_pods):
                    logger.info(f"All {len(running_vllm_pods)} vLLM pods are running")
                    return
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
        
        raise Exception("vLLM pods failed to reach Running status after 30 minutes")


    def test_vllm_api(self):
        logger.info("Testing vLLM API...")
        endpoint = run(
            "kubectl get ingress vllm-deepseek-32b-lws-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
        ).stdout.strip()
        logger.info(f"vLLM API endpoint: {endpoint}")

        if not endpoint:
            raise Exception("Failed to get vLLM API endpoint from ingress")

        self._test_completions_api(endpoint)
        self._test_chat_completions_api(endpoint)
        logger.info("All vLLM API tests passed successfully")
    

    def _test_completions_api(self, endpoint):
        logger.info("Testing completions API...")
        result = run(
            f"""curl -X POST http://{endpoint}/v1/completions \
                -H "Content-Type: application/json" \
                -d '{{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "prompt": "Hello, how are you?", "max_tokens": 50, "temperature": 0.7}}'
            """
        )
        assert '"object":"text_completion"' in result.stdout, "vLLM completions API test failed"
        logger.info("Completions API test passed")
    

    def _test_chat_completions_api(self, endpoint):
        logger.info("Testing chat completions API...")
        result = run(
            f"""curl -X POST http://{endpoint}/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d '{{"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "messages": [{{"role": "user", "content": "What are the benefits of using FSx Lustre with EKS?"}}], "max_tokens": 100, "temperature": 0.7}}'
            """
        )
        assert '"object":"chat.completion"' in result.stdout, "vLLM chat completions API test failed"
        logger.info("Chat completions API test passed")