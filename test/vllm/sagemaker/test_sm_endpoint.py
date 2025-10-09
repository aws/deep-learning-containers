import json
import sagemaker
import time
import boto3
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker import serializers

# Fixed parameters
AWS_REGION = "us-west-2"
INSTANCE_TYPE = "ml.g5.12xlarge"
ROLE = "SageMakerRole"


def get_secret_hf_token():
    secret_name = "test/hf_token"
    region_name = "us-west-2"

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e

    response = json.loads(get_secret_value_response["SecretString"])
    return response


def deploy_endpoint(name, image_uri, role, instance_type):
    try:
        response = get_secret_hf_token()
        hf_token = response.get("HF_TOKEN")
        model = Model(
            name=name,
            image_uri=image_uri,
            role=role,
            env={
                "SM_VLLM_MODEL": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "SM_VLLM_HF-TOKEN": hf_token,
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

        print(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

        print(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)

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


def test_vllm_on_sagemaker(image_uri, endpoint_name):
    if not deploy_endpoint(endpoint_name, image_uri, ROLE, INSTANCE_TYPE):
        print("Failed to deploy endpoint. Exiting.")
        return

    if not wait_for_endpoint(endpoint_name):
        print("Endpoint failed to become ready. Cleaning up...")
        delete_endpoint(endpoint_name)
        return

    test_prompt = "Write a python script to calculate square of n"

    print("Sending request to endpoint...")
    response = invoke_endpoint(
        endpoint_name=endpoint_name, prompt=test_prompt, max_tokens=2400, temperature=0.01
    )

    if response:
        print("\nResponse from endpoint:")
        if isinstance(response, (dict, list)):
            print(json.dumps(response, indent=2))
        else:
            print(response)

        print("\nCleaning up resources...")
        if delete_endpoint(endpoint_name):
            print("Cleanup completed successfully")
        else:
            print("Cleanup failed")
    else:
        print("No response received from the endpoint.")
        print("\nCleaning up resources due to failed inference...")
        delete_endpoint(endpoint_name)
