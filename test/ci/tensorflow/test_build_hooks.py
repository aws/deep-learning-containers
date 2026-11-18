"""Tests for the TensorFlow artifact-cache build hooks (scripts/ci/build/tensorflow/).

The hooks run before and after every TensorFlow image build. pre_build.sh decides
cache hit or miss from S3; post_build.sh uploads a freshly built artifact on a
miss. These tests run the real hooks against the real image configs with a fake
`aws` CLI backed by a temp directory, so each cache path is exercised without S3
or a two-hour build.

Requires bash, sha256sum and mikefarah yq (v4) on PATH, as on the CI runners.
"""

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
HOOKS = REPO / "scripts/ci/build/tensorflow"
CONFIGS = sorted((REPO / ".github/config/image/tensorflow").glob("*.yml"))
BUCKET = "dlc-cicd-wheels"

FAKE_AWS = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    # Minimal `aws s3 cp/ls` backed by $FAKE_S3/<bucket>/<key>.
    [[ "${FAKE_S3_FAIL:-}" == "1" ]] && { echo "fake aws: S3 unavailable" >&2; exit 1; }
    [[ "$1" == s3 ]] || exit 2
    op=$2; shift 2
    include=""; args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --recursive|--only-show-errors|--quiet) shift ;;
        --exclude) shift 2 ;;
        --include) include="$2"; shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    path() { echo "$FAKE_S3/${1#s3://}"; }
    case "$op" in
      ls) [[ -e "$(path "${args[0]}")" ]] ;;
      cp)
        if [[ "${args[0]}" == s3://* ]]; then
          src=$(path "${args[0]}"); [[ -d "$src" ]] || exit 0
          shopt -s nullglob
          for f in "$src"/$include; do cp "$f" "${args[1]}"; done
        else
          dst=$(path "${args[1]}"); mkdir -p "$(dirname "$dst")"; cp "${args[0]}" "$dst"
        fi ;;
      *) exit 2 ;;
    esac
    """
)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _is_source_build(cfg):
    return cfg["metadata"]["device_type"] == "gpu" and bool(
        cfg["build"].get("cuda_compute_capabilities")
    )


SOURCE_BUILD_CONFIGS = [c for c in CONFIGS if _is_source_build(_load(c))]
UPSTREAM_CONFIGS = [c for c in CONFIGS if not _is_source_build(_load(c))]


def _ids(paths):
    return [p.stem for p in paths]


def _artifact_name(cfg):
    if cfg["metadata"]["job_type"] == "training":
        return "tensorflow-2.21.0-cp312-cp312-manylinux_2_27_x86_64.whl"
    return "tensorflow_model_server"


def _prebuilt_dir(cfg):
    return REPO / Path(cfg["build"]["dockerfile"]).parent / "prebuilt"


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    aws = bin_dir / "aws"
    aws.write_text(FAKE_AWS)
    aws.chmod(0o755)
    s3 = tmp_path / "s3"
    s3.mkdir()
    environ = dict(os.environ)
    environ.update(
        PATH=f"{bin_dir}{os.pathsep}{environ['PATH']}",
        FAKE_S3=str(s3),
        GITHUB_ENV=str(tmp_path / "github_env"),
    )
    for var in ("TF_CACHE_HIT", "TF_EXPORT_DIR", "TF_ARTIFACT_PREFIX", "WHEELS_BUCKET"):
        environ.pop(var, None)
    yield environ
    if environ.get("TF_EXPORT_DIR"):
        shutil.rmtree(environ["TF_EXPORT_DIR"], ignore_errors=True)
    for cfg_path in SOURCE_BUILD_CONFIGS:
        for f in _prebuilt_dir(_load(cfg_path)).iterdir():
            if f.name != ".gitkeep":
                f.unlink()


def _run(hook, cfg_path, environ):
    return subprocess.run(
        ["bash", str(HOOKS / hook), "--config-file", str(cfg_path.relative_to(REPO))],
        cwd=REPO,
        env=environ,
        capture_output=True,
        text=True,
    )


def _github_env(environ):
    path = Path(environ["GITHUB_ENV"])
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _prefix(cfg_path, environ):
    out = subprocess.run(
        ["bash", str(HOOKS / "lib/cache_prefix.sh"), "--config-file", str(cfg_path)],
        cwd=REPO,
        env=environ,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _put(environ, key):
    path = Path(environ["FAKE_S3"]) / BUCKET / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("artifact")


@pytest.mark.parametrize("cfg_path", UPSTREAM_CONFIGS, ids=_ids(UPSTREAM_CONFIGS))
def test_upstream_binary_configs_skip_the_cache(cfg_path, env):
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    assert _github_env(env) == {}


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_empty_cache_is_a_miss_with_a_fresh_export_dir(cfg_path, env):
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    out = _github_env(env)
    env["TF_EXPORT_DIR"] = out.get("TF_EXPORT_DIR", "")
    assert out["TF_CACHE_HIT"] == "false"
    assert out["USE_PREBUILT_WHEEL"] == "0"
    assert out["EXPORT_TARGETS"] == f"tf-artifact-export:{out['TF_EXPORT_DIR']}"
    assert os.listdir(out["TF_EXPORT_DIR"]) == []


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_cached_artifact_is_a_hit(cfg_path, env):
    cfg = _load(cfg_path)
    _put(env, _prefix(cfg_path, env) + _artifact_name(cfg))
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    out = _github_env(env)
    assert out["TF_CACHE_HIT"] == "true"
    assert out["USE_PREBUILT_WHEEL"] == "1"
    assert "EXPORT_TARGETS" not in out
    assert (_prebuilt_dir(cfg) / _artifact_name(cfg)).is_file()


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_artifact_of_the_other_job_type_is_not_a_hit(cfg_path, env):
    cfg = _load(cfg_path)
    other = (
        "tensorflow_model_server"
        if cfg["metadata"]["job_type"] == "training"
        else "tensorflow-2.21.0-cp312-cp312-manylinux_2_27_x86_64.whl"
    )
    _put(env, _prefix(cfg_path, env) + other)
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    out = _github_env(env)
    env["TF_EXPORT_DIR"] = out.get("TF_EXPORT_DIR", "")
    assert out["TF_CACHE_HIT"] == "false"


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_leftover_prebuilt_file_is_not_a_hit(cfg_path, env):
    cfg = _load(cfg_path)
    (_prebuilt_dir(cfg) / _artifact_name(cfg)).write_text("stale")
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    out = _github_env(env)
    env["TF_EXPORT_DIR"] = out.get("TF_EXPORT_DIR", "")
    assert out["TF_CACHE_HIT"] == "false"
    assert not (_prebuilt_dir(cfg) / _artifact_name(cfg)).exists()
    assert (_prebuilt_dir(cfg) / ".gitkeep").exists()


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_s3_failure_is_a_miss(cfg_path, env):
    env["FAKE_S3_FAIL"] = "1"
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    out = _github_env(env)
    env["TF_EXPORT_DIR"] = out.get("TF_EXPORT_DIR", "")
    assert out["TF_CACHE_HIT"] == "false"


def test_two_cached_wheels_fail_the_build(env):
    training = [c for c in SOURCE_BUILD_CONFIGS if _load(c)["metadata"]["job_type"] == "training"]
    if not training:
        pytest.skip("no source-built training config")
    cfg_path = training[0]
    prefix = _prefix(cfg_path, env)
    _put(env, prefix + "tensorflow-2.21.0-cp312-cp312-linux_x86_64.whl")
    _put(env, prefix + "tensorflow-2.21.0-cp312-cp312-manylinux_2_27_x86_64.whl")
    result = _run("pre_build.sh", cfg_path, env)
    assert result.returncode != 0
    assert "expected one" in result.stderr


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_miss_uploads_the_exported_artifact(cfg_path, env):
    cfg = _load(cfg_path)
    assert _run("pre_build.sh", cfg_path, env).returncode == 0
    out = _github_env(env)
    env.update(out)
    artifacts = Path(out["TF_EXPORT_DIR"]) / "artifacts"
    artifacts.mkdir()
    (artifacts / _artifact_name(cfg)).write_text("built")
    result = _run("post_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    key = Path(env["FAKE_S3"]) / BUCKET / (out["TF_ARTIFACT_PREFIX"] + _artifact_name(cfg))
    assert key.read_text() == "built"
    assert not Path(out["TF_EXPORT_DIR"]).exists()


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_miss_without_an_exported_artifact_uploads_nothing(cfg_path, env):
    assert _run("pre_build.sh", cfg_path, env).returncode == 0
    env.update(_github_env(env))
    result = _run("post_build.sh", cfg_path, env)
    assert result.returncode == 0, result.stderr
    assert not any(p.is_file() for p in Path(env["FAKE_S3"]).rglob("*"))


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_cache_key_changes_with_each_build_input(cfg_path, env, tmp_path):
    cfg = _load(cfg_path)
    base = _prefix(cfg_path, env)
    commit_key = "tf_commit" if cfg["metadata"]["job_type"] == "training" else "tf_serving_commit"
    changes = {
        "cuda_compute_capabilities": cfg["build"]["cuda_compute_capabilities"] + ",sm_103",
        "bazel_version": "0.0.0",
        "hermetic_cudnn_version": "0.0.0",
        commit_key: "0" * 40,
    }
    for key, value in changes.items():
        changed = dict(cfg, build=dict(cfg["build"], **{key: value}))
        path = tmp_path / f"{key}.yml"
        path.write_text(yaml.safe_dump(changed))
        assert _prefix(path, env) != base, f"cache key ignores {key}"


@pytest.mark.parametrize("cfg_path", SOURCE_BUILD_CONFIGS, ids=_ids(SOURCE_BUILD_CONFIGS))
def test_cache_key_covers_the_source_build_stage(cfg_path, env, tmp_path):
    cfg = _load(cfg_path)
    base = _prefix(cfg_path, env)
    dockerfile = REPO / cfg["build"]["dockerfile"]
    text = dockerfile.read_text()
    edited_dockerfile = tmp_path / "Dockerfile"

    def key_for(dockerfile_text):
        edited_dockerfile.write_text(dockerfile_text)
        path = tmp_path / "cfg.yml"
        path.write_text(
            yaml.safe_dump(dict(cfg, build=dict(cfg["build"], dockerfile=str(edited_dockerfile))))
        )
        return _prefix(path, env)

    assert key_for(text) == base
    assert (
        key_for(
            text.replace(
                "DEBIAN_FRONTEND=noninteractive", "DEBIAN_FRONTEND=noninteractive\nARG X=1", 1
            )
        )
        != base
    )
    # A comment-only edit must not force a rebuild.
    assert key_for(text.replace("\nCOPY ", "\n# note\nCOPY ", 1)) == base
