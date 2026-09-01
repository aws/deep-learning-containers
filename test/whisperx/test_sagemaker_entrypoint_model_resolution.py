"""Unit tests for the SageMaker entrypoint's model-source resolution ladder.

CPU-only, no container: these drive the model-resolution block extracted from the
real ``scripts/docker/whisperx/sagemaker_entrypoint.sh`` (so the test stays coupled
to the shipped script, not a copy) and assert the env it exports.

The ladder mirrors the vLLM SageMaker entrypoint:

  1. WHISPERX_DEFAULT_MODEL already set  -> explicit override wins.
  2. /opt/ml/model populated             -> serve it (WHISPERX_DEFAULT_MODEL=<dir>);
                                            faster-whisper loads a directory path
                                            directly via its os.path.isdir branch.
  3. neither                             -> unset, so server.py uses its default.

The block reads/writes a directory we substitute for /opt/ml/model, so it runs
unchanged on any host.
"""

import re
import subprocess
from pathlib import Path

ENTRYPOINT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "docker"
    / "whisperx"
    / "sagemaker_entrypoint.sh"
)


def _extract_resolution_block() -> str:
    """Return the `if [ -d /opt/ml/model ] ... fi` block from the real entrypoint.

    Keeps the test bound to the shipped script: if the block is edited, the test
    exercises the edit. Fails loudly if the block can't be located (script moved).
    """
    text = ENTRYPOINT.read_text()
    m = re.search(r"^if \[ -d /opt/ml/model \].*?^fi$", text, re.MULTILINE | re.DOTALL)
    assert m, f"model-resolution block not found in {ENTRYPOINT}"
    return m.group(0)


def _run_ladder(model_dir: str, preset_model: str | None) -> dict[str, str]:
    """Run the real resolution block with /opt/ml/model -> model_dir; return env.

    We rewrite the literal path `/opt/ml/model` to the test dir, preset
    WHISPERX_DEFAULT_MODEL if requested, run the block, then print the two vars the
    block manages so the test can assert on them.
    """
    block = _extract_resolution_block().replace("/opt/ml/model", model_dir)
    preset = f'export WHISPERX_DEFAULT_MODEL="{preset_model}"\n' if preset_model is not None else ""
    script = (
        "set -eu\n"
        + preset
        + block
        + "\n"
        + 'echo "WDM=${WHISPERX_DEFAULT_MODEL:-<unset>}"\n'
        + 'echo "HFH=${HF_HOME:-<unset>}"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    env = {}
    for line in out.splitlines():
        if line.startswith(("WDM=", "HFH=")):
            k, v = line.split("=", 1)
            env[k] = v
    return env


def test_populated_dir_no_override_serves_dir(tmp_path):
    """A populated model dir with no override -> WHISPERX_DEFAULT_MODEL points at it."""
    (tmp_path / "model.bin").write_text("stub")
    env = _run_ladder(str(tmp_path), preset_model=None)
    assert env["WDM"] == str(tmp_path)
    assert env["HFH"] == str(tmp_path)


def test_explicit_override_is_respected(tmp_path):
    """A populated dir does NOT clobber an explicit WHISPERX_DEFAULT_MODEL."""
    (tmp_path / "model.bin").write_text("stub")
    env = _run_ladder(str(tmp_path), preset_model="tiny")
    assert env["WDM"] == "tiny"  # override wins over auto-detect
    assert env["HFH"] == str(tmp_path)  # HF_HOME still repointed


def test_empty_dir_falls_through_to_default(tmp_path):
    """An empty /opt/ml/model (SageMaker mounts one even with no ModelDataUrl) ->
    neither var is set, so server.py uses its built-in default."""
    env = _run_ladder(str(tmp_path), preset_model=None)
    assert env["WDM"] == "<unset>"
    assert env["HFH"] == "<unset>"


def test_resolution_block_exists():
    """Guard: the block the tests extract must exist in the shipped entrypoint."""
    assert "WHISPERX_DEFAULT_MODEL=/opt/ml/model" in ENTRYPOINT.read_text()
