"""Integration test for vLLM-Omni SageMaker Bidirectional Streaming (WebSocket).

Exercises the SageMaker bidi WS bridge added to omni_sagemaker_serve.py: a client
calls InvokeEndpointWithBidirectionalStream with model_invocation_path pointing at
vLLM-Omni's native WebSocket route /v1/audio/speech/stream, and receives streamed
TTS audio back over the same connection.

This complements test_sm_omni_endpoint.py (which covers the HTTP /invocations path).
It deploys a real SageMaker endpoint from the DLC image and tears it down in a
finally block, the same lifecycle pattern as the HTTP tests.

Client transport: the experimental async HTTP/2 SDK
(aws_sdk_sagemaker_runtime_http2), NOT boto3 — boto3 has no bidi API. Pinned in
test/vllm-omni/sagemaker/requirements.txt.
"""

import asyncio
import json
import logging
import os

import pytest
from sagemaker.core.resources import Endpoint, EndpointConfig, Model
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant
from test_utils import clean_string, random_suffix_name
from test_utils.constants import INFERENCE_AMI_VERSION, SAGEMAKER_ROLE
from test_utils.huggingface_helper import get_hf_token

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

REGION = "us-west-2"
BIDI_ENDPOINT_URI = f"https://runtime.sagemaker.{REGION}.amazonaws.com:8443"

# The vLLM-Omni WebSocket route we bridge to. Passed to the bidi API SLASHLESS
# (a leading slash fails the model_invocation_path validation with HTTP 400);
# SageMaker's Sidecar prepends the slash so the container sees the real route.
SPEECH_STREAM_PATH = "v1/audio/speech/stream"


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


@pytest.fixture(scope="function")
def bidi_endpoint(aws_session, image_uri, model_id, instance_type):
    """Deploy a real-time endpoint for bidirectional WebSocket streaming.

    Same deploy/teardown lifecycle as the HTTP endpoint tests. The DLC image
    already carries the com.amazonaws.sagemaker.capabilities.bidirectional-streaming
    label (set in Dockerfile.amzn2023), which is what makes SageMaker open a
    WebSocket to the container.
    """
    cleaned_id = clean_string(model_id.split("/")[1], "_./")
    endpoint_name = random_suffix_name(f"vllm-omni-bidi-{cleaned_id}", 50)
    model_name = endpoint_name

    hf_token = get_hf_token(aws_session)
    env = {"SM_VLLM_MODEL": model_id, "HF_TOKEN": hf_token}
    role_arn = aws_session.resolve_role_arn(SAGEMAKER_ROLE)

    model = endpoint_config = endpoint = None
    try:
        LOGGER.info(f"Creating model: {model_name}")
        model = Model.create(
            model_name=model_name,
            primary_container=ContainerDefinition(image=image_uri, environment=env),
            execution_role_arn=role_arn,
        )

        LOGGER.info(f"Creating endpoint config: {endpoint_name}")
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=endpoint_name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=model_name,
                    initial_instance_count=1,
                    instance_type=instance_type,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )

        LOGGER.info(f"Deploying endpoint: {endpoint_name}")
        endpoint = Endpoint.create(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_name,
        )
        endpoint.wait_for_status("InService", timeout=1800)

        yield endpoint
    finally:
        _cleanup([endpoint, endpoint_config, model])


def _make_bidi_client(aws_session):
    """Construct the experimental async HTTP/2 SageMaker runtime client.

    Imported lazily so a missing/renamed experimental SDK fails this test only,
    not collection of the whole sagemaker suite.
    """
    from aws_sdk_sagemaker_runtime_http2.client import AsyncSageMakerRuntimeHTTP2Client
    from aws_sdk_sagemaker_runtime_http2.config import Config, SigV4AuthScheme
    from smithy_aws_core.identity import EnvironmentCredentialsResolver

    scheme = SigV4AuthScheme(service="sagemaker")
    config = Config(
        endpoint_uri=BIDI_ENDPOINT_URI,
        region=aws_session.region,
        auth_schemes={scheme.scheme_id: scheme},
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    return AsyncSageMakerRuntimeHTTP2Client(config=config)


async def _send_json(stream, obj):
    """Send one JSON message as its own COMPLETE UTF8 PayloadPart.

    Each logical message must be its own COMPLETE part: SageMaker collapses a
    PARTIAL...COMPLETE run into a single reassembled WebSocket message, so
    sending session.config / input.text / input.done as separate COMPLETE parts
    keeps them as distinct frames to the handler.
    """
    from aws_sdk_sagemaker_runtime_http2.models import (
        RequestPayloadPart,
        RequestStreamEventPayloadPart,
    )

    part = RequestPayloadPart(
        bytes_=json.dumps(obj).encode("utf-8"),
        data_type="UTF8",
        completion_state="COMPLETE",
    )
    await stream.input_stream.send(RequestStreamEventPayloadPart(value=part))


async def _stream_tts(client, endpoint_name, deadline_s=180):
    """Open a bidi WS to /v1/audio/speech/stream, request TTS, collect audio.

    Returns dict: {audio_bytes, saw_start, saw_done, error}. Reads with a single
    overall deadline — do NOT wrap receive() in per-frame asyncio.wait_for, which
    cancels the in-flight read and corrupts the HTTP/2 stream.
    """
    from aws_sdk_sagemaker_runtime_http2.models import (
        InvokeEndpointWithBidirectionalStreamInput,
        ResponseStreamEventPayloadPart,
    )

    inp = InvokeEndpointWithBidirectionalStreamInput(
        endpoint_name=endpoint_name,
        model_invocation_path=SPEECH_STREAM_PATH,  # slashless
    )

    audio_bytes = 0
    saw_start = saw_done = False
    error = None
    loop = asyncio.get_event_loop()

    async with await client.invoke_endpoint_with_bidirectional_stream(input=inp) as stream:
        _, receiver = await stream.await_output()

        await _send_json(
            stream,
            {
                "type": "session.config",
                "voice": "vivian",
                "language": "English",
                "response_format": "wav",
            },
        )
        await _send_json(
            stream, {"type": "input.text", "text": "Hello, this is a bidirectional streaming test."}
        )
        await _send_json(stream, {"type": "input.done"})

        start = loop.time()
        while loop.time() - start < deadline_s:
            remaining = deadline_s - (loop.time() - start)
            try:
                event = await asyncio.wait_for(receiver.receive(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if event is None:
                break
            if not isinstance(event, ResponseStreamEventPayloadPart):
                # ModelStreamError / InternalStreamFailure / unknown
                error = f"non-payload event: {type(event).__name__}: {event!r}"
                break

            raw = event.value.bytes_ or b""
            data_type = (event.value.data_type or "").upper()
            parsed = None
            if data_type == "UTF8":
                try:
                    parsed = json.loads(raw.decode("utf-8"))
                except Exception:
                    parsed = None

            if parsed is not None:
                mtype = parsed.get("type")
                if mtype == "audio.start":
                    saw_start = True
                elif mtype == "audio.chunk":
                    audio_bytes += (len(parsed.get("audio_b64") or "") * 3) // 4
                elif mtype == "audio.done":
                    saw_done = True
                elif mtype == "session.done":
                    break
                elif mtype == "error":
                    error = parsed.get("message")
                    break
            else:
                # binary audio frame (response_format=wav, non-chunked path)
                audio_bytes += len(raw)

    return {
        "audio_bytes": audio_bytes,
        "saw_start": saw_start,
        "saw_done": saw_done,
        "error": error,
    }


@pytest.mark.parametrize("instance_type", ["ml.g6.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_bidi_speech_stream(bidi_endpoint, aws_session):
    """Stream TTS audio over the SageMaker Bidirectional Streaming WebSocket API.

    Proves the full bridge chain: SageMaker bidi transport -> the WS branch of
    SageMakerRouteMiddleware -> vLLM-Omni's /v1/audio/speech/stream handler ->
    streamed audio back to the client.
    """

    # The experimental HTTP/2 SDK reads creds via EnvironmentCredentialsResolver,
    # so surface the test session's credentials into the environment.
    creds = aws_session.session.get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    os.environ.setdefault("AWS_REGION", aws_session.region)

    client = _make_bidi_client(aws_session)

    # First request also pays torch.compile + CUDA graph warmup, which can be
    # slow; the 180s stream deadline absorbs it. One retry covers a transient
    # first-call warmup that overruns.
    result = None
    for attempt in range(2):
        result = asyncio.run(_stream_tts(client, bidi_endpoint.endpoint_name))
        LOGGER.info(f"Bidi TTS attempt {attempt + 1}: {result}")
        if result["saw_start"] and result["audio_bytes"] > 2000 and not result["error"]:
            break

    assert result["error"] is None, f"bidi stream returned an error frame: {result['error']}"
    assert result["saw_start"], "never received audio.start — bridge did not reach the TTS handler"
    assert result["audio_bytes"] > 2000, f"streamed audio too small: {result['audio_bytes']} bytes"
    LOGGER.info(f"Bidi speech-stream test PASSED — {result['audio_bytes']} audio bytes streamed")


@pytest.mark.parametrize("instance_type", ["ml.g6.xlarge"], indirect=True)
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"], indirect=True)
def test_vllm_omni_bidi_default_path_rejected(bidi_endpoint, aws_session):
    """Omitting model_invocation_path targets /invocations-bidirectional-stream,
    which the middleware rejects (no vLLM-Omni handler there). The handshake must
    fail rather than opening a usable socket."""
    from aws_sdk_sagemaker_runtime_http2.models import (
        InvokeEndpointWithBidirectionalStreamInput,
        ResponseStreamEventPayloadPart,
    )

    creds = aws_session.session.get_credentials().get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
    if creds.token:
        os.environ["AWS_SESSION_TOKEN"] = creds.token
    os.environ.setdefault("AWS_REGION", aws_session.region)

    client = _make_bidi_client(aws_session)

    async def _probe():
        # No model_invocation_path -> default /invocations-bidirectional-stream.
        inp = InvokeEndpointWithBidirectionalStreamInput(endpoint_name=bidi_endpoint.endpoint_name)
        try:
            async with await client.invoke_endpoint_with_bidirectional_stream(input=inp) as stream:
                _, receiver = await stream.await_output()
                await _send_json(stream, {"type": "session.config", "voice": "vivian"})
                event = await asyncio.wait_for(receiver.receive(), timeout=30)
                # A refused handshake yields no usable payload frame.
                if event is None or not isinstance(event, ResponseStreamEventPayloadPart):
                    return True
                return False  # got a real frame -> NOT rejected
        except Exception as e:
            LOGGER.info(f"default-path connect refused as expected: {type(e).__name__}: {e}")
            return True

    assert asyncio.run(_probe()), "default bidi path should be rejected by the middleware"
    LOGGER.info("Bidi default-path reject test PASSED")
