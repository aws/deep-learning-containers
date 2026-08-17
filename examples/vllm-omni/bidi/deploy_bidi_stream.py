"""Bidirectional WebSocket streaming on a vLLM-Omni SageMaker endpoint.

Deploys the vLLM-Omni SageMaker DLC, opens a SageMaker Bidirectional Streaming
WebSocket to the omni TTS route (`/v1/audio/speech/stream`), streams a request,
collects the audio, and tears the endpoint down.

The image carries `LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true`,
which is what makes SageMaker open a WebSocket to the container.

Transport: boto3 has no bidirectional API — this uses the experimental
`aws_sdk_sagemaker_runtime_http2` client (needs Python >= 3.12). Install it with:

    pip install "sagemaker>=3.0.0" boto3 \
      "git+https://github.com/awslabs/aws-sdk-python.git#subdirectory=clients/aws-sdk-sagemaker-runtime-http2"

Fill in EXECUTION_ROLE_ARN (a SageMaker execution role in your account) before running.
"""

import asyncio
import json
import os
import time

import boto3
from sagemaker.core.resources import Endpoint, EndpointConfig, Model  # sagemaker>=3.0.0
from sagemaker.core.shapes import ContainerDefinition, ProductionVariant

# --- Configuration -----------------------------------------------------------
REGION = "us-west-2"
IMAGE_URI = "763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:omni-sagemaker-cuda-v1"
EXECUTION_ROLE_ARN = "arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
INSTANCE_TYPE = "ml.g6.xlarge"

# REQUIRED for bidirectional streaming: this AMI carries the SageMaker Sidecar
# that terminates the client's HTTP/2 WebSocket and bridges it to the container's
# WS route. Omit it and SageMaker never opens the socket to the container.
INFERENCE_AMI_VERSION = "al2023-ami-sagemaker-inference-gpu-4-1"

# The omni WebSocket route. Passed to the bidi API SLASHLESS — a leading slash
# fails model_invocation_path validation (HTTP 400); SageMaker's Sidecar prepends
# the slash so the container sees the real route.
SPEECH_STREAM_PATH = "v1/audio/speech/stream"
BIDI_ENDPOINT_URI = f"https://runtime.sagemaker.{REGION}.amazonaws.com:8443"


def make_bidi_client():
    from aws_sdk_sagemaker_runtime_http2.client import AsyncSageMakerRuntimeHTTP2Client
    from aws_sdk_sagemaker_runtime_http2.config import Config, SigV4AuthScheme
    from smithy_aws_core.identity import EnvironmentCredentialsResolver

    # The experimental HTTP/2 SDK signs with EnvironmentCredentialsResolver, which
    # reads ONLY the AWS_* env vars. boto3's default chain resolves creds wherever
    # they live (profile, SSO, instance role, or env vars already set); surface them
    # into the environment so this works regardless of how you're authenticated.
    # Without this, a profile/SSO/role user would deploy, wait for InService, then
    # fail to authenticate the stream call.
    creds = boto3.Session().get_credentials()
    if creds is None:
        raise RuntimeError("No AWS credentials found — configure a profile, SSO, or role first.")
    frozen = creds.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    os.environ.setdefault("AWS_REGION", REGION)

    scheme = SigV4AuthScheme(service="sagemaker")
    return AsyncSageMakerRuntimeHTTP2Client(
        config=Config(
            endpoint_uri=BIDI_ENDPOINT_URI,
            region=REGION,
            auth_schemes={scheme.scheme_id: scheme},
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
    )


async def _send_json(stream, obj):
    # Each logical message is its own COMPLETE UTF8 part: SageMaker collapses a
    # PARTIAL...COMPLETE run into one reassembled WebSocket message, so
    # session.config / input.text / input.done stay distinct frames.
    from aws_sdk_sagemaker_runtime_http2.models import (
        RequestPayloadPart,
        RequestStreamEventPayloadPart,
    )

    part = RequestPayloadPart(
        bytes_=json.dumps(obj).encode("utf-8"), data_type="UTF8", completion_state="COMPLETE"
    )
    await stream.input_stream.send(RequestStreamEventPayloadPart(value=part))


async def stream_tts(endpoint_name, text, deadline_s=180):
    from aws_sdk_sagemaker_runtime_http2.models import (
        InvokeEndpointWithBidirectionalStreamInput,
        ResponseStreamEventPayloadPart,
    )

    client = make_bidi_client()
    inp = InvokeEndpointWithBidirectionalStreamInput(
        endpoint_name=endpoint_name, model_invocation_path=SPEECH_STREAM_PATH
    )
    audio_bytes, saw_start, saw_done, error = 0, False, False, None
    loop = asyncio.get_running_loop()

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
        await _send_json(stream, {"type": "input.text", "text": text})
        await _send_json(stream, {"type": "input.done"})

        start = loop.time()
        # One overall deadline — do NOT wrap receive() in a per-frame wait_for,
        # which cancels the in-flight read and corrupts the HTTP/2 stream.
        while loop.time() - start < deadline_s:
            try:
                event = await asyncio.wait_for(
                    receiver.receive(), timeout=deadline_s - (loop.time() - start)
                )
            except asyncio.TimeoutError:
                break
            if event is None:
                break
            if not isinstance(event, ResponseStreamEventPayloadPart):
                error = f"non-payload event: {type(event).__name__}"
                break
            raw = event.value.bytes_ or b""
            if (event.value.data_type or "").upper() == "UTF8":
                msg = json.loads(raw.decode("utf-8"))
                t = msg.get("type")
                if t == "audio.start":
                    saw_start = True
                elif t == "audio.chunk":
                    audio_bytes += (len(msg.get("audio_b64") or "") * 3) // 4
                elif t == "audio.done":
                    saw_done = True
                elif t == "session.done":
                    break
                elif t == "error":
                    error = msg.get("message")
                    break
            else:
                audio_bytes += len(raw)

    return {
        "audio_bytes": audio_bytes,
        "saw_start": saw_start,
        "saw_done": saw_done,
        "error": error,
    }


def main():
    name = f"vllm-omni-bidi-{int(time.time())}"
    # Every SageMaker call passes region=REGION explicitly: the v3 resource API
    # does not fall back to $AWS_REGION on its own.
    model = endpoint_config = endpoint = None
    try:
        model = Model.create(
            model_name=name,
            primary_container=ContainerDefinition(
                image=IMAGE_URI, environment={"SM_VLLM_MODEL": MODEL_ID}
            ),
            execution_role_arn=EXECUTION_ROLE_ARN,
            region=REGION,
        )
        endpoint_config = EndpointConfig.create(
            endpoint_config_name=name,
            region=REGION,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=name,
                    initial_instance_count=1,
                    instance_type=INSTANCE_TYPE,
                    inference_ami_version=INFERENCE_AMI_VERSION,
                ),
            ],
        )
        endpoint = Endpoint.create(endpoint_name=name, endpoint_config_name=name, region=REGION)
        print(f"Deploying {name} — waiting for InService (~40 min)...")
        endpoint.wait_for_status("InService", timeout=2400)

        result = asyncio.run(
            stream_tts(name, "Hello, this is a bidirectional streaming demo on SageMaker.")
        )
        print("Stream result:", result)
        assert result["error"] is None and result["saw_start"] and result["audio_bytes"] > 2000, (
            result
        )
        print(f"\nStreamed {result['audio_bytes']} audio bytes back over the bidi WebSocket.")
    finally:
        # Best-effort teardown of each resource — a live GPU endpoint bills continuously.
        for label, obj in (
            ("endpoint", endpoint),
            ("endpoint-config", endpoint_config),
            ("model", model),
        ):
            if obj is None:
                continue
            try:
                obj.delete()
                print(f"deleted {label}: {name}")
            except Exception as e:
                print(f"WARNING: failed to delete {label} {name}: {e}")


if __name__ == "__main__":
    main()
