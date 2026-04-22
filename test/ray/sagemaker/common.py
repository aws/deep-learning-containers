"""Shared constants, helpers, fixtures, and test implementations for Ray SageMaker endpoint tests.

CPU and GPU test modules import from here, setting only DEVICE and INSTANCE_TYPE.
Uses boto3 directly for model/endpoint lifecycle (custom DLC containers).
"""

import json
import logging
from pprint import pformat

import pytest
from ray.utils import (
    IRIS_SAMPLES,
    MIN_MNIST_ACCURACY,
    S3_BUCKET,
    S3_PREFIX,
    SENTIMENT_SAMPLES,
    download_all_test_images,
    make_all_digit_pngs,
    make_all_sine_wavs,
    validate_audio_response,
    validate_densenet_response,
    validate_iris_response,
    validate_mnist_response,
    validate_sentiment_response,
)
from sagemaker.serializers import IdentitySerializer
from test_utils import clean_string, random_suffix_name, wait_for_status
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ENDPOINT_WAIT_PERIOD = 60  # seconds between status checks
ENDPOINT_WAIT_LENGTH = 30  # max number of retries
ENDPOINT_INSERVICE = "InService"


def _parse_response(response):
    """Parse SageMaker predictor response to dict/list."""
    return json.loads(response) if isinstance(response, (str, bytes)) else response


def get_endpoint_status(sagemaker_client, endpoint_name):
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    LOGGER.debug(f"Describe endpoint response: {pformat(response)}")
    return response["EndpointStatus"]


def build_models(device):
    """Build the MODELS dict parameterized by device ("cpu" or "gpu").

    Differences driven by device:
      - S3 sub-path: <model>/<device>/model.tar.gz
      - RAYSERVE_NUM_GPUS: "0" for cpu, "1" for gpu (mnist-direct-app only)
    """
    num_gpus = "0" if device == "cpu" else "1"
    return {
        "cv-densenet": {
            "s3_key": f"cv-densenet/{device}/model.tar.gz",
            "env": {},
        },
        "mnist-direct-app": {
            "s3_key": f"mnist-direct-app/{device}/model.tar.gz",
            "env": {"SM_RAYSERVE_APP": "deployment:app", "RAYSERVE_NUM_GPUS": num_gpus},
        },
        "tabular": {
            "s3_key": f"tabular/{device}/model.tar.gz",
            "env": {},
        },
        "nlp": {
            "s3_key": f"nlp/{device}/model.tar.gz",
            "env": {},
        },
        "audio-ffmpeg": {
            "s3_key": f"audio-ffmpeg/{device}/model.tar.gz",
            "env": {},
        },
    }


# ---------------------------------------------------------------------------
# Fixtures — called by the thin CPU/GPU test modules
# ---------------------------------------------------------------------------


def make_model_name_fixture():
    """Create the model_name fixture."""

    @pytest.fixture(scope="function")
    def model_name(request):
        return request.param

    return model_name


def make_model_endpoint_fixture(device, instance_type):
    """Create the model_endpoint fixture using boto3 directly."""
    models = build_models(device)
    prefix = f"ray-{device}-"

    @pytest.fixture(scope="function")
    def model_endpoint(aws_session, image_uri, model_name):
        sagemaker_client = aws_session.sagemaker
        model_config = models[model_name]
        s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_config['s3_key']}"
        cleaned = clean_string(model_name, "_./")
        endpoint_name = random_suffix_name(f"{prefix}{cleaned}", 50)
        sm_model_name = endpoint_name

        LOGGER.info(f"Deploying endpoint: {endpoint_name}")
        LOGGER.info(f"  Image: {image_uri}")
        LOGGER.info(f"  Model data: {s3_uri}")

        sagemaker_client.create_model(
            ModelName=sm_model_name,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": s3_uri,
                "Environment": model_config["env"] or {},
            },
            ExecutionRoleArn=SAGEMAKER_ROLE,
        )

        variant = {
            "VariantName": "AllTraffic",
            "ModelName": sm_model_name,
            "InitialInstanceCount": 1,
            "InstanceType": instance_type,
        }
        if device == "gpu":
            variant["InferenceAmiVersion"] = INFERENCE_AMI_VERSION
            LOGGER.info(f"  Using inference AMI: {INFERENCE_AMI_VERSION}")

        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_name,
            ProductionVariants=[variant],
        )

        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_name,
        )

        LOGGER.info(f"Waiting for endpoint {ENDPOINT_INSERVICE} status...")
        assert wait_for_status(
            ENDPOINT_INSERVICE,
            ENDPOINT_WAIT_PERIOD,
            ENDPOINT_WAIT_LENGTH,
            get_endpoint_status,
            sagemaker_client,
            endpoint_name,
        )

        # Create a lightweight predictor-like object for test compatibility
        from sagemaker.predictor import Predictor

        predictor = Predictor(endpoint_name=endpoint_name)

        yield predictor

        LOGGER.info(f"Deleting endpoint: {endpoint_name}")
        for cleanup_fn, name in [
            (lambda: sagemaker_client.delete_endpoint(EndpointName=endpoint_name), "endpoint"),
            (
                lambda: sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=endpoint_name
                ),
                "endpoint config",
            ),
            (lambda: sagemaker_client.delete_model(ModelName=sm_model_name), "model"),
        ]:
            try:
                cleanup_fn()
            except Exception as e:
                LOGGER.warning(f"Cleanup {name} failed: {e}")

    return model_endpoint


# ---------------------------------------------------------------------------
# Test implementations — called by the thin CPU/GPU test modules
# ---------------------------------------------------------------------------


def run_test_cv_densenet(model_endpoint):
    """DenseNet image classification — both kitten and flower images, validate top-5 structure."""
    images = download_all_test_images()
    model_endpoint.serializer = IdentitySerializer(content_type="image/jpeg")

    for img_name, img_data in images.items():
        response = model_endpoint.predict(img_data)
        result = _parse_response(response)
        LOGGER.info(f"cv-densenet {img_name} response: {result}")

        err = validate_densenet_response(result)
        assert not err, f"cv-densenet {img_name}: {err}"

        top = result["predictions"][0]
        LOGGER.info(
            f"  {img_name} -> {top['class_name']} "
            f"(class_id={top['class_id']}, prob={top['probability']:.4f})"
        )


def run_test_mnist_direct_app(model_endpoint):
    """MNIST via SM_RAYSERVE_APP — classify all 10 digits, enforce accuracy threshold."""
    digit_pngs = make_all_digit_pngs()
    correct = 0
    total = len(digit_pngs)
    model_endpoint.serializer = IdentitySerializer(content_type="image/png")

    for digit_id, png_data in sorted(digit_pngs.items()):
        response = model_endpoint.predict(png_data)
        result = _parse_response(response)
        LOGGER.info(f"mnist-direct-app digit_{digit_id} response: {result}")

        err = validate_mnist_response(result)
        assert not err, f"mnist-direct-app digit_{digit_id}: {err}"

        pred = result["prediction"]
        conf = result["confidence"]
        if pred == digit_id:
            correct += 1
            LOGGER.info(f"  digit_{digit_id} -> predicted={pred} confidence={conf:.4f}")
        else:
            LOGGER.warning(
                f"  digit_{digit_id} -> predicted={pred} expected={digit_id} confidence={conf:.4f}"
            )

    accuracy = int(correct * 100 / total)
    LOGGER.info(f"mnist-direct-app accuracy: {correct}/{total} ({accuracy}%)")
    assert accuracy >= MIN_MNIST_ACCURACY, (
        f"MNIST accuracy {accuracy}% below threshold {MIN_MNIST_ACCURACY}%"
    )


def run_test_tabular(model_endpoint, check_packages=False):
    """Iris classification — 6 samples (2 per species), validate predicted species."""
    for features, expected, desc in IRIS_SAMPLES:
        payload = {"features": features}
        response = model_endpoint.predict(payload)
        result = _parse_response(response)
        LOGGER.info(f"tabular {desc} response: {result}")

        err = validate_iris_response(result)
        assert not err, f"tabular {desc}: {err}"

        pred = result["prediction"]
        conf = result["confidence"]
        assert pred == expected, f"tabular {desc}: predicted '{pred}', expected '{expected}'"
        LOGGER.info(f"  {desc} -> {pred} ({conf:.4f})")

    if check_packages:
        response = model_endpoint.predict({"features": IRIS_SAMPLES[0][0]})
        result = _parse_response(response)
        pkgs = result.get("installed_packages", {})
        assert pkgs, "Expected installed_packages in response (requirements.txt not installed?)"
        LOGGER.info(f"  requirements.txt packages: {pkgs}")


def run_test_nlp(model_endpoint):
    """DistilBERT sentiment — 6 samples (3 pos, 3 neg), validate predicted label."""
    for text, expected_label in SENTIMENT_SAMPLES:
        payload = {"text": text}
        response = model_endpoint.predict(payload)
        result = _parse_response(response)
        LOGGER.info(f"nlp response for '{text}': {result}")

        err = validate_sentiment_response(result)
        assert not err, f"nlp '{text}': {err}"

        label = result["predictions"][0]["label"]
        score = result["predictions"][0]["score"]
        assert label == expected_label, (
            f"nlp '{text}': predicted '{label}', expected '{expected_label}'"
        )
        LOGGER.info(f"  '{text}' -> {label} ({score:.4f})")


def run_test_audio_ffmpeg(model_endpoint):
    """Wav2Vec2 transcription — 3 sine waves, validate structure + ffmpeg backend."""
    wavs = make_all_sine_wavs()
    model_endpoint.serializer = IdentitySerializer(content_type="audio/wav")

    for name, wav_data in wavs:
        response = model_endpoint.predict(wav_data)
        result = _parse_response(response)
        LOGGER.info(f"audio-ffmpeg {name} response: {result}")

        err = validate_audio_response(result, check_ffmpeg_backend=True)
        assert not err, f"audio-ffmpeg {name}: {err}"

        LOGGER.info(
            f'  {name} -> "{result["transcription"]}" (backend: {result.get("audio_backend", "?")})'
        )
