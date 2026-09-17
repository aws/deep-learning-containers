"""Integration test for vLLM-Omni SageMaker endpoint — SageMaker SDK v3"""

import json
import logging
import time
import uuid

import boto3
import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import (
    AsyncInferenceClientConfig,
    AsyncInferenceConfig,
    AsyncInferenceOutputConfig,
    ContainerDefinition,
    ProductionVariant,
)
from test_utils import clean_string, random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token
from test_utils.instance_capacity import (
    build_instance_pools,
    is_capacity_error,
    normalize_instance_types,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

VIDEO_MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
VIDEO_INSTANCE_TYPES = ["ml.g6.2xlarge", "ml.g6.4xlarge", "ml.g5.2xlarge"]

# PEFT LoRA: SD-3.5-medium (~20 GiB peak) fits a single 24 GB GPU with no offload.
# Base is HF-pulled at runtime; the PEFT adapter ships via the model's S3
# model_data, extracted to /opt/ml/model/adapters/sd35-lora.
LORA_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
LORA_ADAPTER_S3 = "s3://dlc-cicd-models/omni-models/sd35-yarnart-peft-adapter.tar.gz"
LORA_ADAPTER_PATH = "/opt/ml/model/adapters/sd35-lora"
LORA_INSTANCE_TYPES = [
    "ml.g6.xlarge",
    "ml.g6.2xlarge",
    "ml.g6.4xlarge",
    "ml.g5.xlarge",
    "ml.g5.2xlarge",
]


@pytest.fixture(scope="function")
def model_id(request):
    return request.param


@pytest.fixture(scope="function")
def instance_type(request):
    return request.param


def _cleanup(resources):
    """Best-effort delete for a list of v3 resource objects (None-safe)."""
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as e:
            LOGGER.warning(f"Cleanup {type(resource).__name__} failed: {e}")


def _create_model(model_name, image_uri, env, role_arn, model_data_url=None):
    """Create a v3 Model resource pointing at the DLC image.

    model_data_url (optional): an S3 .tar.gz SageMaker extracts to /opt/ml/model
    on the endpoint — used to ship a LoRA adapter alongside an HF-pulled base.
    """
    LOGGER.info(f"Creating model: {model_name}")
    container_kwargs = {"image": image_uri, "environment": env}
    if model_data_url:
        container_kwargs["model_data_url"] = model_data_url
    return Model.create(
        model_name=model_name,
        primary_container=ContainerDefinition(**container_kwargs),
        execution_role_arn=role_arn,
    )


@pytest.fixture(scope="function")
def model_endpoint(aws_session, image_uri, model_id, instance_type):
    """Deploy a realtime endpoint over a native SageMaker instance-pool ladder."""
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-{cleaned_id}", 50)
    model_name = endpoint_name

    hf_token = get_hf_token(aws_session)
    env = {"SM_VLLM_MODEL": model_id, "HF_TOKEN": hf_token}
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    model = endpoint_config = endpoint = None
    try:
        model = _create_model(model_name, image_uri, env, role_arn)

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_pools=build_instance_pools(instance_type),
                    variant_instance_provision_timeout_in_seconds=1800,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )

        LOGGER.info(f"Deploying endpoint: {endpoint_name} on {instance_type}")
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        # Leave enough of the runner's credential session for inference and cleanup.
        endpoint.wait_for_status("InService", timeout=2700)

        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


# Every rung is a single 24 GB GPU and can serve this TTS model unaided.
@pytest.mark.parametrize(
    "instance_type",
    [["ml.g6.xlarge", "ml.g6.2xlarge", "ml.g6.4xlarge", "ml.g5.xlarge", "ml.g5.2xlarge"]],
    indirect=True,
)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_endpoint(model_endpoint, aws_session):
    """TTS via /invocations, covering both response transports on ONE endpoint.

    Two SageMaker transports share this deployment to avoid a second ~15-min
    endpoint spin-up (and a second AWS-token-expiry window on the long-running
    sagemaker job):
      1. Buffered — InvokeEndpoint, one audio blob.
      2. Response-streaming — InvokeEndpointWithResponseStream with
         stream=true + response_format=pcm, yielding chunked PCM. This is the
         customer's Alexa transport. (The bidirectional-WebSocket transport is
         covered separately in test_sm_omni_bidi_endpoint.py.)
    """
    endpoint = model_endpoint

    # --- Transport 1: buffered InvokeEndpoint ---
    payload = json.dumps(
        {
            "input": "Hello, this is a test of the text to speech system.",
            "voice": "vivian",
            "language": "English",
        }
    )

    LOGGER.info("Sending buffered TTS request via /invocations with route=/v1/audio/speech")
    # First request triggers torch.compile + CUDA graph capture (~67s),
    # which exceeds SageMaker's 60s invoke timeout. Retry after warmup completes.
    for attempt in range(3):
        try:
            result = endpoint.invoke(
                body=payload,
                content_type="application/json",
                custom_attributes="route=/v1/audio/speech",
            )
            break
        except Exception as e:
            LOGGER.warning(f"Buffered attempt {attempt + 1}/3 failed: {e}")
            if attempt == 2:
                raise
            time.sleep(30)

    audio_bytes = result.body.read()
    LOGGER.info(f"Buffered TTS response: {len(audio_bytes)} bytes")
    assert len(audio_bytes) > 1000, f"buffered TTS output too small: {len(audio_bytes)} bytes"

    # --- Transport 2: response-streaming InvokeEndpointWithResponseStream ---
    # vLLM-Omni's /v1/audio/speech returns a StreamingResponse of raw PCM when
    # the body sets stream=true + pcm; SageMaker surfaces it as an EventStream
    # of PayloadPart frames. boto3 is used (not the v3 SDK): sagemaker-core's
    # invoke_with_response_stream pipes the response through a generic codec
    # with no event-stream branch and raises on a live EventStream; boto3
    # yields the raw EventStream directly.
    smr = aws_session.session.client("sagemaker-runtime", region_name=aws_session.region)
    stream_body = json.dumps(
        {
            "input": "Hello, this is a streaming text to speech test.",
            "voice": "vivian",
            "language": "English",
            "stream": True,
            "response_format": "pcm",
        }
    ).encode()

    LOGGER.info("Sending streaming TTS request via InvokeEndpointWithResponseStream")
    # Model is already warm from transport 1, so a single attempt is enough.
    resp = smr.invoke_endpoint_with_response_stream(
        EndpointName=endpoint.endpoint_name,
        Body=stream_body,
        ContentType="application/json",
        CustomAttributes="route=/v1/audio/speech",
    )
    streamed = bytearray()
    n_chunks = 0
    for event in resp["Body"]:
        if "PayloadPart" in event:
            streamed += event["PayloadPart"]["Bytes"]  # raw PCM — do not decode
            n_chunks += 1
        elif "ModelStreamError" in event:
            raise AssertionError(f"ModelStreamError: {event['ModelStreamError']}")
        elif "InternalStreamFailure" in event:
            raise AssertionError(f"InternalStreamFailure: {event['InternalStreamFailure']}")

    LOGGER.info(f"Streaming TTS response: {len(streamed)} PCM bytes across {n_chunks} chunk(s)")
    # Byte count is the robust assertion; SageMaker/botocore may coalesce
    # PayloadPart frames, so chunk count is logged but not hard-asserted.
    assert len(streamed) > 1000, f"streamed PCM too small: {len(streamed)} bytes"
    LOGGER.info("TTS endpoint test PASSED (buffered + response-streaming)")


@pytest.fixture(scope="function")
def async_endpoint(aws_session, image_uri, model_id, instance_type):
    """Deploy an async endpoint over a native SageMaker instance-pool ladder."""
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-async-{cleaned_id}", 50)
    model_name = endpoint_name
    account_id = aws_session.sts.get_caller_identity()["Account"]
    s3_output = f"s3://sagemaker-{aws_session.region}-{account_id}/vllm-omni-async-output/"

    hf_token = get_hf_token(aws_session)
    env = {"SM_VLLM_MODEL": model_id, "HF_TOKEN": hf_token}
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    model = endpoint_config = endpoint = None
    try:
        try:
            model = _create_model(model_name, image_uri, env, role_arn)

            LOGGER.info(f"Creating async endpoint config: {endpoint_name}")
            endpoint_config = EndpointConfig.create(
                endpoint_config_name=endpoint_name,
                production_variants=[
                    ProductionVariant(
                        variant_name="AllTraffic",
                        model_name=model_name,
                        initial_instance_count=1,
                        instance_pools=build_instance_pools(instance_type),
                        variant_instance_provision_timeout_in_seconds=1800,
                        inference_ami_version=INFERENCE_AMI_VERSION,
                    ),
                ],
                async_inference_config=AsyncInferenceConfig(
                    output_config=AsyncInferenceOutputConfig(s3_output_path=s3_output),
                    client_config=AsyncInferenceClientConfig(
                        max_concurrent_invocations_per_instance=1,
                    ),
                ),
            )

            LOGGER.info(f"Deploying async endpoint: {endpoint_name} on {instance_type}")
            endpoint = Endpoint.create(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_name,
            )
            # Leave enough of the runner's credential session for inference and cleanup.
            endpoint.wait_for_status("InService", timeout=2700)
        except Exception as e:
            if model_id == VIDEO_MODEL_ID and is_capacity_error(e):
                candidates = normalize_instance_types(instance_type)
                pytest.skip(
                    "No SageMaker capacity for video-async after exhausting "
                    f"{len(candidates)} native instance-pool candidates {candidates}. "
                    f"Last error: {e}"
                )
            raise

        yield endpoint, s3_output
    finally:
        _cleanup([endpoint, endpoint_config, model])


@pytest.mark.parametrize(
    "instance_type",
    [["ml.g6.xlarge", "ml.g6.2xlarge", "ml.g6.4xlarge", "ml.g5.xlarge", "ml.g5.2xlarge"]],
    indirect=True,
)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_tts_async_endpoint(async_endpoint):
    """TTS via async inference — no 60s timeout, up to 1 hour."""
    endpoint, s3_output = async_endpoint
    # Build the S3 client at point-of-use, not at fixture setup: the endpoint
    # deploy + InService wait can run ~40 min, long enough for a session token
    # captured earlier to expire before the first S3 call.
    s3_client = boto3.client("s3")

    payload = json.dumps(
        {
            "input": "Hello, this is a test of async text to speech.",
            "voice": "vivian",
            "language": "English",
        }
    )

    LOGGER.info("Sending async TTS request")
    input_location = _upload_payload_to_s3(s3_client, payload, s3_output, endpoint.endpoint_name)
    result = endpoint.invoke_async(
        input_location=input_location,
        content_type="application/json",
        custom_attributes="route=/v1/audio/speech",
    )

    output_location = result.output_location
    LOGGER.info(f"Async output location: {output_location}")

    # Poll for result (up to 5 minutes)
    bucket, key = _parse_s3_uri(output_location)
    for i in range(60):
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            audio_bytes = obj["Body"].read()
            LOGGER.info(f"Async TTS response: {len(audio_bytes)} bytes (after {i * 5}s)")
            assert len(audio_bytes) > 1000, f"TTS output too small: {len(audio_bytes)} bytes"
            LOGGER.info("Async TTS endpoint test PASSED")
            return
        except s3_client.exceptions.NoSuchKey:
            time.sleep(5)

    pytest.fail("Async inference timed out after 300s")


# VACE needs at least 32 GB host RAM, so xlarge pools are intentionally excluded.
@pytest.mark.parametrize(
    "instance_type",
    [VIDEO_INSTANCE_TYPES],
    indirect=True,
)
@pytest.mark.parametrize("model_id", [VIDEO_MODEL_ID], indirect=True)
def test_vllm_omni_video_async_endpoint(async_endpoint):
    """Video gen via async inference + /v1/videos/sync.

    Instance choice (ml.g6.2xlarge): L4 24 GB VRAM (compute 8.9) + 32 GB host
    RAM. Equivalent VRAM/host RAM to g5.2xlarge but with better us-west-2
    capacity. L4 uses PyTorch SDPA fallback (no FA3 dependency since VACE
    diffusion runs through standard attention). Host RAM is the constraint
    that bit us earlier on 16 GB instances during HF model load.

    Async inference removes SageMaker's 60s real-time invoke timeout, so this
    pattern is the recommended way to serve video generation behind a
    SageMaker endpoint. /v1/videos/sync requires multipart/form-data input;
    SageMaker InvokeEndpoint forwards arbitrary ContentType values through to
    the model server, so the client builds the multipart body locally and
    sends it directly — no in-middleware conversion required. Result is
    video/mp4 bytes deposited at S3 output location.
    """
    endpoint, s3_output = async_endpoint
    # Build the S3 client at point-of-use (see note in the TTS async test): the
    # long endpoint-deploy wait can outlive a session token captured earlier.
    s3_client = boto3.client("s3")

    boundary = uuid.uuid4().hex
    payload = _build_multipart_body(
        {
            "prompt": "a dog running on a beach",
            "num_frames": "17",
            "num_inference_steps": "4",
            "size": "480x320",
            "seed": "42",
        },
        boundary,
    )
    content_type = f"multipart/form-data; boundary={boundary}"

    LOGGER.info("Sending async video request via /invocations -> /v1/videos/sync")
    input_location = _upload_payload_to_s3(
        s3_client, payload, s3_output, endpoint.endpoint_name, content_type
    )
    result = endpoint.invoke_async(
        input_location=input_location,
        content_type=content_type,
        custom_attributes="route=/v1/videos/sync",
    )

    output_location = result.output_location
    LOGGER.info(f"Async output location: {output_location}")

    # VACE-1.3B at 4 steps / 17 frames takes ~3s warm + ~3-4 min for first
    # request (model load + torch.compile). Poll up to 10 minutes.
    bucket, key = _parse_s3_uri(output_location)
    for i in range(120):
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=key)
            video_bytes = obj["Body"].read()
            LOGGER.info(f"Async video response: {len(video_bytes)} bytes (after {i * 5}s)")
            assert len(video_bytes) > 1000, f"video output too small: {len(video_bytes)} bytes"
            # Sanity: content-type metadata should be video/mp4 (set by
            # /v1/videos/sync handler).
            content_type = obj.get("ContentType", "")
            LOGGER.info(f"Output content-type: {content_type}")
            LOGGER.info("Async video endpoint test PASSED")
            return
        except s3_client.exceptions.NoSuchKey:
            time.sleep(5)

    pytest.fail("Async video inference timed out after 600s")


@pytest.fixture(scope="function")
def lora_endpoint(aws_session, image_uri, instance_type):
    """Deploy a realtime endpoint with a PEFT LoRA adapter registered at startup.

    Base (SD-3.5-medium) is HF-pulled at runtime; the adapter ships as the
    model's S3 model_data, extracted to /opt/ml/model/adapters/sd35-lora.
    """
    endpoint_name = random_suffix_name("vllm-omni-lora-sd35", 50)
    model_name = endpoint_name

    hf_token = get_hf_token(aws_session)
    env = {
        "SM_VLLM_MODEL": LORA_MODEL_ID,
        "HF_TOKEN": hf_token,
        "SM_VLLM_ENABLE_LORA": "true",
        "SM_VLLM_LORA_MODULES": json.dumps({"name": "sd35-lora", "path": LORA_ADAPTER_PATH}),
        "SM_VLLM_MAX_LORA_RANK": "64",
        "SM_VLLM_TRUST_REMOTE_CODE": "true",
    }
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    model = endpoint_config = endpoint = None
    try:
        model = _create_model(model_name, image_uri, env, role_arn, model_data_url=LORA_ADAPTER_S3)

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_pools=build_instance_pools(instance_type),
                    variant_instance_provision_timeout_in_seconds=1800,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )

        LOGGER.info(f"Deploying LoRA endpoint: {endpoint_name} on {instance_type}")
        endpoint = Endpoint.create(endpoint_name=endpoint_name, endpoint_config_name=endpoint_name)
        endpoint.wait_for_status("InService", timeout=2700)

        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


def _invoke_image(endpoint, body):
    """POST an image-gen request via /invocations; return the base64 image string."""
    # First request pays model load + warmup; retry past SageMaker's 60s invoke timeout.
    for attempt in range(3):
        try:
            result = endpoint.invoke(
                body=json.dumps(body),
                content_type="application/json",
                custom_attributes="route=/v1/images/generations",
            )
            break
        except Exception as e:
            LOGGER.warning(f"Image invoke attempt {attempt + 1}/3 failed: {e}")
            if attempt == 2:
                raise
            time.sleep(30)
    data = json.loads(result.body.read())
    b64 = (data.get("data") or [{}])[0].get("b64_json")
    assert b64 and len(b64) > 1000, f"no/small image in response: {str(data)[:300]}"
    return b64


# SD-3.5-medium fits every rung (single 24 GB GPU, no offload).
@pytest.mark.parametrize("instance_type", [LORA_INSTANCE_TYPES], indirect=True)
def test_vllm_omni_peft_lora_endpoint(lora_endpoint):
    """PEFT LoRA per-request selection on /v1/images/generations.

    Generates the SAME prompt/seed twice — without, then with, the per-request
    `lora` field. A real bind check: the images must DIFFER, proving the adapter
    is applied per request (not merely loaded at startup).
    """
    endpoint = lora_endpoint
    base = {"prompt": "a cat sitting on a chair", "size": "512x512", "seed": 42, "n": 1}

    LOGGER.info("Image gen WITHOUT adapter (baseline)")
    img_base = _invoke_image(endpoint, base)

    LOGGER.info("Image gen WITH per-request lora field")
    img_lora = _invoke_image(
        endpoint,
        {**base, "lora": {"name": "sd35-lora", "local_path": LORA_ADAPTER_PATH, "scale": 1.0}},
    )

    assert img_base != img_lora, (
        "LoRA had no effect: with-adapter output identical to baseline "
        "(per-request `lora` field not applied by the middleware/server)"
    )
    LOGGER.info("PEFT LoRA endpoint test PASSED (adapter applied per-request)")


def _upload_payload_to_s3(
    s3_client, payload, s3_output, endpoint_name, content_type="application/json"
):
    """Upload request payload to S3 for async inference."""
    bucket, prefix = _parse_s3_uri(s3_output)
    suffix = "bin" if not content_type.startswith("application/json") else "json"
    key = f"{prefix}{endpoint_name}-input.{suffix}"
    body = payload if isinstance(payload, (bytes, bytearray)) else payload.encode()
    s3_client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
    return f"s3://{bucket}/{key}"


def _build_multipart_body(data: dict, boundary: str) -> bytes:
    """Build a multipart/form-data body from a dict of string values."""
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
        for k, v in data.items()
    ]
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode()


def _parse_s3_uri(uri):
    """Parse s3://bucket/key into (bucket, key)."""
    parts = uri.replace("s3://", "").split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""
