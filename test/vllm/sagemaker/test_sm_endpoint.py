import json
import sagemaker
import time
import boto3
import argparse
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker import serializers

AWS_REGION = "us-west-2"
INSTANCE_TYPE = "ml.g5.12xlarge"


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy and test SageMaker endpoint")
    parser.add_argument(
        "--name", type=str, required=True, help="Name for the endpoint, config, and model"
    )
    parser.add_argument(
        "--image-uri", type=str, required=True, help="ECR image URI for the model container"
    )
    parser.add_argument(
        "--role", type=str, default="SageMakerRole", help="SageMaker execution role"
    )
    parser.add_argument("--region", type=str, default=AWS_REGION, help="AWS region")
    parser.add_argument(
        "--instance-type", type=str, default=INSTANCE_TYPE, help="SageMaker instance type"
    )
    return parser.parse_args()


def deploy_endpoint(name, image_uri, role, instance_type):
    try:
        model = Model(
            name=name,
            image_uri=image_uri,
            role=role,
            env={
                "SM_VLLM_MODEL": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            },
        )

        endpoint_config = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=name,
            wait=True,
        )
        return True
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        return False


def invoke_endpoint(endpoint_name, prompt, max_tokens=2400, temperature=0.01):
    try:
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=serializers.JSONSerializer(),
        )

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
        }

        response = predictor.predict(payload)

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


def delete_endpoint(endpoint_name):
    try:
        sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)

        # Delete the endpoint
        print(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

        # Delete the endpoint configuration
        print(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

        # Delete the model
        print(f"Deleting model: {endpoint_name}")
        sagemaker_client.delete_model(ModelName=endpoint_name)

        print("Successfully deleted all resources")
        return True
    except Exception as e:
        print(f"Error during deletion: {str(e)}")
        return False


def wait_for_endpoint(endpoint_name, timeout=1800):
    sagemaker_client = boto3.client("sagemaker", region_name=AWS_REGION)
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            if status == "InService":
                return True
            elif status in ["Failed", "OutOfService"]:
                print(f"Endpoint creation failed with status: {status}")
                return False

            print(f"Endpoint status: {status}. Waiting...")
            time.sleep(30)
        except Exception as e:
            print(f"Error checking endpoint status: {str(e)}")
            return False

    print("Timeout waiting for endpoint to be ready")
    return False


def main():
    args = parse_args()

    if not deploy_endpoint(args.name, args.image_uri, args.role, args.instance_type):
        print("Failed to deploy endpoint. Exiting.")
        return

    if not wait_for_endpoint(args.name, args.region):
        print("Endpoint failed to become ready. Cleaning up...")
        delete_endpoint(args.name, args.region)
        return

    test_prompt = "Write a python script to calculate square of n"

    print("Sending request to endpoint...")
    response = invoke_endpoint(
        endpoint_name=args.name, prompt=test_prompt, max_tokens=2400, temperature=0.01
    )

    if response:
        print("\nResponse from endpoint:")
        if isinstance(response, (dict, list)):
            print(json.dumps(response, indent=2))
        else:
            print(response)

        print("\nCleaning up resources...")
        if delete_endpoint(args.name, args.region):
            print("Cleanup completed successfully")
        else:
            print("Cleanup failed")
    else:
        print("No response received from the endpoint.")
        print("\nCleaning up resources due to failed inference...")
        delete_endpoint(args.name, args.region)


if __name__ == "__main__":
    main()
