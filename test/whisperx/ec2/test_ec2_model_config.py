# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""EC2 model-launch-config tests for the WhisperX ASR DLC (GPU).

Group A: these tests validate how the server's launch-time env vars shape the
model-resolution contract — WHISPERX_DEFAULT_MODEL (the pinned/served model),
WHISPERX_SERVED_MODEL_NAME (a client-facing alias), and
WHISPERX_ALLOW_MODEL_OVERRIDE (per-request on-demand loading). Each case needs a
container launched with different env, so unlike test_ec2_gpu.py they do NOT use
the shared default `container` fixture; instead each spins up (and tears down) a
custom-env container via common.run_container_with_env.

WHISPERX_DEFAULT_MODEL=tiny is used throughout so the startup warm-load (and any
on-demand override load) finishes in seconds — tiny is a valid faster-whisper
model that downloads far faster than the large-v2 default.

Assertions target the resolution contract only (served model id, 200 vs 404, and
the 404 error-body shape) — never exact transcript text, which is nondeterministic.
"""

import requests
from whisperx.ec2.common import (
    AUDIO_EN,
    download_fixture,
    post_transcription,
    run_container_with_env,
)

DEVICE = "gpu"
DOCKER_RUN_FLAGS = ["--gpus", "all"]
# Small, fast-downloading Whisper model so warm-load and on-demand override load
# take seconds, not the minutes large-v2 needs. tiny is a valid faster-whisper id.
FAST_MODEL = "tiny"


def test_custom_default_model(image_uri, aws_session, tmp_path):
    """WHISPERX_DEFAULT_MODEL pins the served id; only that id (or none) resolves.

    Launches with WHISPERX_DEFAULT_MODEL=tiny and asserts /v1/models advertises
    `tiny`, that omitting `model` and naming `tiny` both transcribe (200), and
    that a different real model (`large-v2`) is rejected 404 (override off).
    """
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    with run_container_with_env(
        image_uri, {"WHISPERX_DEFAULT_MODEL": FAST_MODEL}, device=DEVICE, flags=DOCKER_RUN_FLAGS
    ) as c:
        port = c["port"]

        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data[0]["id"] == FAST_MODEL, f"expected served id {FAST_MODEL!r}, got {data}"

        # Omitted model -> resolves to the pinned default -> 200 with real text.
        resp = post_transcription(port, audio, response_format="json")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("text", "").strip(), "expected non-empty text with default model"

        # Naming the pinned model explicitly -> 200.
        resp = post_transcription(port, audio, model=FAST_MODEL, response_format="json")
        assert resp.status_code == 200, resp.text

        # A different real model with override off -> 404.
        resp = post_transcription(port, audio, model="large-v2", response_format="json")
        assert resp.status_code == 404, resp.text


def test_served_model_alias(image_uri, aws_session, tmp_path):
    """WHISPERX_SERVED_MODEL_NAME exposes an alias; the 404 detail names that alias.

    Launches tiny behind the alias `whisper-1` and asserts /v1/models reports
    `whisper-1`, that both the alias and the real backing id `tiny` transcribe
    (200), and that an unknown id is rejected 404 with a detail that names the
    served alias (not the internal `tiny`).
    """
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    env = {"WHISPERX_DEFAULT_MODEL": FAST_MODEL, "WHISPERX_SERVED_MODEL_NAME": "whisper-1"}
    with run_container_with_env(image_uri, env, device=DEVICE, flags=DOCKER_RUN_FLAGS) as c:
        port = c["port"]

        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data[0]["id"] == "whisper-1", f"expected alias id 'whisper-1', got {data}"

        # The alias resolves to the pinned model -> 200.
        resp = post_transcription(port, audio, model="whisper-1", response_format="json")
        assert resp.status_code == 200, resp.text

        # The real backing model id still resolves too -> 200.
        resp = post_transcription(port, audio, model=FAST_MODEL, response_format="json")
        assert resp.status_code == 200, resp.text

        # An unknown id -> 404 whose detail advertises the served alias.
        resp = post_transcription(port, audio, model="whisper-large", response_format="json")
        assert resp.status_code == 404, resp.text
        detail = resp.json().get("detail", "")
        assert "whisper-1" in detail, f"404 detail should name the served alias: {detail!r}"


def test_model_override_allowed(image_uri, aws_session, tmp_path):
    """WHISPERX_ALLOW_MODEL_OVERRIDE=true loads an unlisted model on demand.

    Launches tiny with override enabled and asserts that naming a *different*
    valid model (`base`) transcribes 200 — loaded on demand — rather than 404.
    Kept to `base` (small, fast) so the on-demand load stays quick.
    """
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    env = {"WHISPERX_DEFAULT_MODEL": FAST_MODEL, "WHISPERX_ALLOW_MODEL_OVERRIDE": "true"}
    with run_container_with_env(image_uri, env, device=DEVICE, flags=DOCKER_RUN_FLAGS) as c:
        # `base` is not the pinned model, but override loads it on demand -> 200.
        resp = post_transcription(c["port"], audio, model="base", response_format="json")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("text", "").strip(), "expected non-empty text from override model"


def test_model_override_denied_response(image_uri, tmp_path):
    """Override off: an unknown model yields a 404 whose body names the served id.

    Asserts the error *shape* (FastAPI's {"detail": ...}), not just the status:
    the detail must say the model "does not exist" and name the served id
    (`tiny`). The model is resolved before the upload is read, so a present but
    unread dummy file part is enough — no S3 audio needed.
    """
    dummy = tmp_path / "dummy.wav"
    dummy.write_bytes(b"RIFF0000WAVE")
    with run_container_with_env(
        image_uri, {"WHISPERX_DEFAULT_MODEL": FAST_MODEL}, device=DEVICE, flags=DOCKER_RUN_FLAGS
    ) as c:
        resp = post_transcription(c["port"], str(dummy), model="nonexistent-model")
        assert resp.status_code == 404, resp.text
        detail = resp.json().get("detail", "")
        assert "does not exist" in detail, f"expected 'does not exist' in 404 detail: {detail!r}"
        assert FAST_MODEL in detail, f"expected served id {FAST_MODEL!r} in 404 detail: {detail!r}"
