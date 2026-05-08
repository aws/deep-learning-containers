"""Shared constants, helpers, fixtures, and test implementations for Ray SageMaker endpoint tests.

CPU and GPU test modules import from here, setting only DEVICE and INSTANCE_TYPE.
Uses SageMaker SDK v3 core resources (Model/EndpointConfig/Endpoint) for resource lifecycle.
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
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import clean_string, random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _parse_response(response):
    """Parse raw endpoint response body to dict/list."""
    return json.loads(response) if isinstance(response, (str, bytes)) else response


def _cleanup(resources):
    """Best-effort delete for a list of v3 resource objects (None-safe)."""
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _invoke(endpoint, data, content_type="application/json"):
    """Invoke endpoint, returning raw response bytes. Handles JSON serialization."""
    if content_type == "application/json" and not isinstance(data, (str, bytes)):
        body = json.dumps(data)
    else:
        body = data
    result = endpoint.invoke(body=body, content_type=content_type)
    return result.body.read()


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
    """Create the model_endpoint fixture using v3 core resources."""
    models = build_models(device)
    prefix = f"ray-{device}-"

    @pytest.fixture(scope="function")
    def model_endpoint(aws_session, image_uri, model_name):
        model_config = models[model_name]
        s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{model_config['s3_key']}"
        cleaned = clean_string(model_name, "_./")
        endpoint_name = random_suffix_name(f"{prefix}{cleaned}", 50)
        sm_model_name = endpoint_name

        LOGGER.info(f"Deploying endpoint: {endpoint_name}")
        LOGGER.info(f"  Image: {image_uri}")
        LOGGER.info(f"  Model data: {s3_uri}")

        role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)
        model = endpoint_config = endpoint = None
        try:
            model = Model.create(
                model_name=sm_model_name,
                primary_container=ContainerDefinition(
                    image=image_uri,
                    model_data_url=s3_uri,
                    environment=model_config["env"] or {},
                ),
                execution_role_arn=role_arn,
            )

            variant_kwargs = dict(
                variant_name="AllTraffic",
                model_name=sm_model_name,
                initial_instance_count=1,
                instance_type=instance_type,
            )
            if device == "gpu":
                variant_kwargs["inference_ami_version"] = INFERENCE_AMI_VERSION
                LOGGER.info(f"  Using inference AMI: {INFERENCE_AMI_VERSION}")

            endpoint_config = EndpointConfig.create(
                endpoint_config_name=endpoint_name,
                production_variants=[ProductionVariant(**variant_kwargs)],
            )

            endpoint = Endpoint.create(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_name,
            )
            endpoint.wait_for_status("InService")

            yield endpoint
        finally:
            _cleanup([endpoint, endpoint_config, model])

    return model_endpoint


# ---------------------------------------------------------------------------
# Test implementations — called by the thin CPU/GPU test modules
# ---------------------------------------------------------------------------


def run_test_cv_densenet(endpoint):
    """DenseNet image classification — both kitten and flower images, validate top-5 structure."""
    images = download_all_test_images()

    for img_name, img_data in images.items():
        response = _invoke(endpoint, img_data, content_type="image/jpeg")
        result = _parse_response(response)
        LOGGER.info(f"cv-densenet {img_name} response: {result}")

        err = validate_densenet_response(result)
        assert not err, f"cv-densenet {img_name}: {err}"

        top = result["predictions"][0]
        LOGGER.info(
            f"  {img_name} -> {top['class_name']} "
            f"(class_id={top['class_id']}, prob={top['probability']:.4f})"
        )


def run_test_mnist_direct_app(endpoint):
    """MNIST via SM_RAYSERVE_APP — classify all 10 digits, enforce accuracy threshold."""
    digit_pngs = make_all_digit_pngs()
    correct = 0
    total = len(digit_pngs)

    for digit_id, png_data in sorted(digit_pngs.items()):
        response = _invoke(endpoint, png_data, content_type="image/png")
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


def run_test_tabular(endpoint, check_packages=False):
    """Iris classification — 6 samples (2 per species), validate predicted species."""
    for features, expected, desc in IRIS_SAMPLES:
        response = _invoke(endpoint, {"features": features})
        result = _parse_response(response)
        LOGGER.info(f"tabular {desc} response: {result}")

        err = validate_iris_response(result)
        assert not err, f"tabular {desc}: {err}"

        pred = result["prediction"]
        conf = result["confidence"]
        assert pred == expected, f"tabular {desc}: predicted '{pred}', expected '{expected}'"
        LOGGER.info(f"  {desc} -> {pred} ({conf:.4f})")

    if check_packages:
        response = _invoke(endpoint, {"features": IRIS_SAMPLES[0][0]})
        result = _parse_response(response)
        pkgs = result.get("installed_packages", {})
        assert pkgs, "Expected installed_packages in response (requirements.txt not installed?)"
        LOGGER.info(f"  requirements.txt packages: {pkgs}")


def run_test_nlp(endpoint):
    """DistilBERT sentiment — 6 samples (3 pos, 3 neg), validate predicted label."""
    for text, expected_label in SENTIMENT_SAMPLES:
        response = _invoke(endpoint, {"text": text})
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


def run_test_audio_ffmpeg(endpoint):
    """Wav2Vec2 transcription — 3 sine waves, validate structure + ffmpeg backend."""
    wavs = make_all_sine_wavs()

    for name, wav_data in wavs:
        response = _invoke(endpoint, wav_data, content_type="audio/wav")
        result = _parse_response(response)
        LOGGER.info(f"audio-ffmpeg {name} response: {pformat(result)}")

        err = validate_audio_response(result, check_ffmpeg_backend=True)
        assert not err, f"audio-ffmpeg {name}: {err}"

        LOGGER.info(
            f'  {name} -> "{result["transcription"]}" (backend: {result.get("audio_backend", "?")})'
        )
