import json
import logging
import sys
import time

import pytest
from sagemaker.huggingface import HuggingFaceModel

logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

CPU_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "ml.m5.xlarge"),
]

GPU_MODELS = [
    ("BAAI/bge-m3", "ml.g6.12xlarge"),
    ("sentence-transformers/all-MiniLM-L6-v2", "ml.g6.12xlarge"),
]

DEPLOY_TIMEOUT = 1800


def _run(image_uri: str, role: str, model_id: str, instance_type: str) -> None:
    endpoint_name = model_id.replace("/", "-").replace(".", "-")[:40]
    endpoint_name = f"{endpoint_name}-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"
    model = HuggingFaceModel(
        name=endpoint_name,
        env={"HF_MODEL_ID": model_id},
        role=role,
        image_uri=image_uri,
    )
    predictor = None
    try:
        predictor = model.deploy(
            instance_type=instance_type,
            initial_instance_count=1,
            endpoint_name=endpoint_name,
            container_startup_health_check_timeout=DEPLOY_TIMEOUT,
        )
        logging.info(f"Endpoint {endpoint_name} deployed.")

        output = predictor.predict({"inputs": "What is Deep Learning?"})
        logging.info(f"Output: {json.dumps(output)[:500]}")
        assert output, f"Empty embedding response for {model_id}"
    finally:
        if predictor is not None:
            predictor.delete_model()
            predictor.delete_endpoint()


def test_embedding_inference(image_uri, role_arn, device_type):
    models = GPU_MODELS if device_type == "gpu" else CPU_MODELS
    for model_id, instance_type in models:
        logging.info(f"Testing {model_id} on {instance_type} ({device_type})")
        _run(image_uri, role_arn, model_id, instance_type)
