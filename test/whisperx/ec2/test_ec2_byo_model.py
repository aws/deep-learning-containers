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
"""EC2 bring-your-own-model test for the WhisperX ASR DLC (GPU).

Proves a customer-supplied model (staged from S3, mounted into the container) is
actually served — not silently ignored in favor of a Hub download.

The proof is offline-enforced and cannot false-pass:

  * HF_HUB_OFFLINE=1 blocks all Hugging Face network access, so any served model
    MUST come from the mounted directory, never a download.
  * WHISPERX_DEFAULT_MODEL is set to the mounted directory PATH. faster-whisper's
    WhisperModel.__init__ takes an `os.path.isdir(...)` branch that loads a local
    directory directly, bypassing model-name -> HF-repo resolution. So the ct2
    files in the mounted dir are what load.
  * Negative control: the SAME offline launch WITHOUT the mount must fail to become
    healthy (no model present, download blocked). This proves the positive test
    would actually catch a broken dir-load rather than passing regardless.

The model is a NON-default (large-v3; the image default is large-v2) so a pass is
not attributable to the baked/default path.

Staging (run once, see .claude/handoff/whisperx-byo-model-verification.md):
  A flat tarball of Systran/faster-whisper-large-v3 ct2 files lives at
  s3://dlc-cicd-models/whisperx-models/faster-whisper-large-v3.tar.gz. On the test
  host it is downloaded + extracted to a directory whose path is exported as
  WHISPERX_BYO_MODEL_DIR. Absent that env var the tests skip (no model staged).

VAD + diarization still load offline (pyannote VAD ships in the whisperx wheel;
diarization loads from the baked local path), so the container boots under
HF_HUB_OFFLINE=1.
"""

import os

import pytest
import requests
from whisperx.ec2.common import (
    AUDIO_EN,
    download_fixture,
    post_transcription,
    run_container_with_env,
    start_container,
    stop_container,
    wait_for_health,
)

DEVICE = "gpu"
GPU_FLAGS = ["--gpus", "all"]

# Directory holding the extracted non-default model (staged from S3 by the CI
# workflow / devbox script). The container sees it at /opt/ml/model.
BYO_MODEL_DIR = os.environ.get("WHISPERX_BYO_MODEL_DIR")
CONTAINER_MODEL_PATH = "/opt/ml/model"

pytestmark = pytest.mark.skipif(
    not BYO_MODEL_DIR,
    reason="WHISPERX_BYO_MODEL_DIR not set; stage the non-default model dir to enable (see handoff doc)",
)


def _mount_flags():
    """GPU flags + a read-only bind mount of the staged model dir at /opt/ml/model."""
    return GPU_FLAGS + ["-v", f"{BYO_MODEL_DIR}:{CONTAINER_MODEL_PATH}:ro"]


def test_offline_serves_mounted_model(image_uri, aws_session, tmp_path):
    """A mounted non-default model serves with the Hub blocked -> it loaded from the dir.

    HF_HUB_OFFLINE=1 makes a Hub download impossible, and WHISPERX_DEFAULT_MODEL
    points at the mounted directory (faster-whisper's isdir branch loads it
    directly). A 200 with non-empty text therefore proves the mounted model served.
    """
    audio = download_fixture(aws_session, AUDIO_EN, str(tmp_path / AUDIO_EN))
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WHISPERX_DEFAULT_MODEL": CONTAINER_MODEL_PATH,
    }
    with run_container_with_env(image_uri, env, device=DEVICE, flags=_mount_flags()) as c:
        port = c["port"]

        # /v1/models advertises the served id — here the mounted path, since the
        # request `model` field is ignored and the launch value is what's served.
        resp = requests.get(f"http://localhost:{port}/v1/models", timeout=10)
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"][0]["id"] == CONTAINER_MODEL_PATH, resp.text

        # The real proof: offline + mounted dir -> a real transcription.
        resp = post_transcription(port, audio, response_format="json")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("text", "").strip(), (
            "empty text: the mounted model did not serve under HF_HUB_OFFLINE=1"
        )


def test_offline_without_mount_fails_health(image_uri):
    """Negative control: offline + NO mounted model must never become healthy.

    Same offline launch as the positive test but without the volume mount and
    pointing at an (absent) path. The lifespan hook warm-loads the model BEFORE
    uvicorn binds (server.py) and does not guard that call, so with no local model
    and the Hub blocked the load raises, startup aborts, and the socket is never
    served. /ping therefore never reaches 200. This proves the positive test's
    pass is attributable to the mount, not to a Hub fallback that would make it
    pass regardless.
    """
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WHISPERX_DEFAULT_MODEL": CONTAINER_MODEL_PATH,  # path won't exist without the mount
    }
    container_id, port = start_container(image_uri, DEVICE, docker_run_flags=GPU_FLAGS, env=env)
    try:
        # A correct container fails startup fast (no model to resolve, download
        # blocked) and never serves /ping. Bounded wait: only confirm it does NOT
        # become healthy; the failure mode is a refused socket, not a slow boot.
        with pytest.raises(TimeoutError):
            wait_for_health(port=port, timeout=90)
    finally:
        stop_container(container_id)
