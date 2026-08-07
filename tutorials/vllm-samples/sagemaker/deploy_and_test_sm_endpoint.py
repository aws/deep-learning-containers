import argparse
import json
import os
import sagemaker
from sagemaker.model import Model
from sagemaker import serializers
from sagemaker.predictor import Predictor


def deploy_endpoint(
    endpoint_name, container_uri, iam_role, instance_type, model_id, hf_token
):
    """Deploy vLLM model to SageMaker endpoint"""
    try:
        print(f"Starting deployment of endpoint: {endpoint_name}")
        print(f"Using image: {container_uri}")
        print(f"Instance type: {instance_type}")

        print("Creating SageMaker model...")
        model = Model(
            name=endpoint_name,
            image_uri=container_uri,
            role=iam_role,
            env={
                "SM_VLLM_MODEL": model_id,  # Model to load
                "SM_VLLM_HF_TOKEN": hf_token,  # HuggingFace token for model access
            },
        )
        print("Model created successfully")
        print("Starting endpoint deployment (this may take 10-15 minutes)...")

        model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            wait=True,  # Wait for deployment to complete
        )
        print(f"Endpoint {endpoint_name} deployed successfully")
        return True
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False


def cleanup_endpoint(endpoint_name):
    """Delete SageMaker endpoint and model"""
    try:
        import boto3

        sagemaker_client = boto3.client("sagemaker")

        print(f"Cleaning up endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        sagemaker_client.delete_model(ModelName=endpoint_name)
        print(f"Endpoint {endpoint_name} cleaned up successfully")
        return True
    except Exception as e:
        print(f"Cleanup failed: {str(e)}")
        return False


def invoke_endpoint(endpoint_name, prompt, max_tokens=2400, temperature=0.01):
    """Invoke SageMaker endpoint with vLLM model for text generation"""
    try:
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=serializers.JSONSerializer(),
        )

        payload = {
            "messages": [{"role": "user", "content": prompt}],  # Chat format
            "max_tokens": max_tokens,  # Response length limit
            "temperature": temperature,  # Randomness (0=deterministic, 1=creative)
            "top_p": 0.9,  # Nucleus sampling
            "top_k": 50,  # Top-k sampling
        }

        response = predictor.predict(payload)

        # Handle different response formats
        if isinstance(response, bytes):
            response = response.decode("utf-8")

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                print("Warning: Response is not valid JSON. Returning as string.")

        return response

    except Exception as e:
        print(f"Inference failed: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description="SageMaker vLLM Inference")
    parser.add_argument(
        "--endpoint-name", required=True, help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--container-uri",
        help="DLC image URI",
        default=os.getenv(
            "CONTAINER_URI",
            "763104351884.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.2-gpu-py312",
        ),
    )
    parser.add_argument(
        "--iam-role", help="IAM role ARN", default=os.getenv("IAM_ROLE")
    )
    parser.add_argument(
        "--instance-type", default="ml.g5.12xlarge", help="Instance type"
    )
    parser.add_argument(
        "--model-id",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--hf-token", help="HuggingFace token", default=os.getenv("HF_TOKEN", "")
    )
    parser.add_argument(
        "--prompt",
        default="Write a python code to generate n prime numbers",
        help="Inference prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=2400, help="Maximum tokens")
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Sampling temperature"
    )

    args = parser.parse_args()

    if not args.iam_role:
        print("Error: IAM role required")
        return

    # Deploy endpoint
    if not deploy_endpoint(
        args.endpoint_name,
        args.container_uri,
        args.iam_role,
        args.instance_type,
        args.model_id,
        args.hf_token,
    ):
        return

    # Run inference
    print("\nSending request to endpoint...")
    response = invoke_endpoint(
        endpoint_name=args.endpoint_name,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if response:
        print("\nResponse from endpoint:")
        if isinstance(response, (dict, list)):
            print(json.dumps(response, indent=2))
        else:
            print(response)
    else:
        print("No response received from the endpoint.")

    # Cleanup
    print("\nCleaning up resources...")
    cleanup_endpoint(args.endpoint_name)


if __name__ == "__main__":
    main()
