"""Unit tests for the SageMaker entrypoint's SM_VLLM_* -> argv translation.

CPU-only, no container: these load the shipped
``scripts/docker/vllm/sagemaker_args.py`` directly (so the tests stay coupled to the
file baked into the image, not a copy) and assert on the argv list it builds.

The contract under test, which mirrors vLLM's own config-file translation in
``vllm/utils/argparse_utils.py`` (list -> one argv token per element, dict -> a single
JSON token):

  * A JSON array expands to one argv token per element, so flags declared
    ``nargs="+"`` (--lora-modules, --served-model-name, --custom-ops, ... every
    list-typed config field) can receive more than one value.
  * A single JSON object stays exactly one token, so JSON-valued flags such as
    --speculative-config keep working.
  * Anything that is not JSON is passed through untouched, so chat templates and
    paths containing spaces are never split.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SM_ARGS = Path(__file__).resolve().parents[2] / "scripts" / "docker" / "vllm" / "sagemaker_args.py"
ENTRYPOINTS = [
    Path(__file__).resolve().parents[2] / "scripts" / "docker" / "vllm" / name
    for name in ("sagemaker_entrypoint.sh", "omni_sagemaker_entrypoint.sh")
]


def _load_module():
    """Import the shipped helper by path so the test exercises the real file."""
    spec = importlib.util.spec_from_file_location("sagemaker_args", SM_ARGS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sagemaker_args = _load_module()


def build(env, model_dir="/nonexistent"):
    """Build argv from env, defaulting model_dir to a path that never exists."""
    return sagemaker_args.build_args(env, model_dir=model_dir)


def value_of(argv, flag):
    """Return the argv tokens that follow `flag`, up to the next flag."""
    assert flag in argv, f"{flag} not in {argv}"
    start = argv.index(flag) + 1
    values = []
    for token in argv[start:]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


# --- the reported bug: multiple LoRA modules -------------------------------------


def test_json_array_of_objects_becomes_one_token_per_adapter():
    """V2323183393: `[{...},{...},{...}]` must reach vLLM as three argv tokens."""
    adapters = [
        {"name": "translation-v1", "path": "/opt/ml/model/loras/translation-v1"},
        {"name": "translation-v2", "path": "/opt/ml/model/loras/translation-v2"},
        {"name": "summarisation-v1", "path": "/opt/ml/model/loras/summarisation-v1"},
    ]
    argv = build({"SM_VLLM_LORA_MODULES": json.dumps(adapters)})
    tokens = value_of(argv, "--lora-modules")
    assert len(tokens) == 3
    # Each token must independently satisfy vLLM's LoRAParserAction: json.loads to a
    # mapping it can splat into LoRAModulePath(**d).
    assert [json.loads(t) for t in tokens] == adapters


def test_whitespace_separated_json_objects_become_separate_tokens():
    """The format copied from vLLM's CLI docs: `'{...}' '{...}'` in one env var."""
    argv = build(
        {"SM_VLLM_LORA_MODULES": '{"name":"a","path":"/loras/a"} {"name":"b","path":"/loras/b"}'}
    )
    tokens = value_of(argv, "--lora-modules")
    assert [json.loads(t) for t in tokens] == [
        {"name": "a", "path": "/loras/a"},
        {"name": "b", "path": "/loras/b"},
    ]


def test_single_lora_object_still_one_token():
    """Single-adapter deployments worked before the fix and must keep working."""
    value = '{"name":"translation-v1","path":"/opt/ml/model/loras/translation-v1"}'
    argv = build({"SM_VLLM_LORA_MODULES": value})
    assert value_of(argv, "--lora-modules") == [value]


def test_name_equals_path_pairs_are_not_split():
    """`a=/p` is vLLM's old single-adapter format; it has no multi-value spelling,
    so it must pass through untouched rather than be guessed at."""
    argv = build({"SM_VLLM_LORA_MODULES": "translation-v1=/loras/translation-v1"})
    assert value_of(argv, "--lora-modules") == ["translation-v1=/loras/translation-v1"]


# --- the other ~20 nargs="+" flags ----------------------------------------------


def test_json_array_of_strings_expands_for_list_typed_flags():
    """--served-model-name is nargs='+' like every list-typed config field."""
    argv = build({"SM_VLLM_SERVED_MODEL_NAME": '["primary","alias"]'})
    assert value_of(argv, "--served-model-name") == ["primary", "alias"]


def test_json_array_of_ints_expands_to_string_tokens():
    argv = build({"SM_VLLM_CUDAGRAPH_CAPTURE_SIZES": "[1, 2, 4]"})
    assert value_of(argv, "--cudagraph-capture-sizes") == ["1", "2", "4"]


def test_empty_json_array_omits_the_flag():
    """A flag with zero values is an argparse error; drop it as vLLM's config
    loader does (`if value:` before appending the flag)."""
    argv = build({"SM_VLLM_CUSTOM_OPS": "[]"})
    assert "--custom-ops" not in argv


# --- flags that must keep a JSON array as ONE token -----------------------------


@pytest.mark.parametrize(
    "env_key,flag",
    [
        ("SM_VLLM_ALLOWED_ORIGINS", "--allowed-origins"),
        ("SM_VLLM_ALLOWED_METHODS", "--allowed-methods"),
        ("SM_VLLM_ALLOWED_HEADERS", "--allowed-headers"),
    ],
)
def test_json_loads_typed_flags_keep_array_as_single_token(env_key, flag):
    """vLLM strips nargs from these and gives them type=json.loads, so the array
    itself is the value (cli_args.py _customize_cli_kwargs)."""
    argv = build({env_key: '["https://a.example","https://b.example"]'})
    assert value_of(argv, flag) == ['["https://a.example","https://b.example"]']


def test_middleware_keeps_value_as_single_token():
    """--middleware is action='append': one value per occurrence, never nargs."""
    argv = build({"SM_VLLM_MIDDLEWARE": "my_module.MyMiddleware"})
    assert value_of(argv, "--middleware") == ["my_module.MyMiddleware"]


# --- JSON-object flags: the V2245238423 regression guard ------------------------


def test_single_json_object_value_is_preserved_verbatim():
    """--speculative-config takes one JSON object; quoting/spacing must survive."""
    value = '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'
    argv = build({"SM_VLLM_SPECULATIVE_CONFIG": value})
    assert value_of(argv, "--speculative-config") == [value]


def test_nested_json_object_is_not_split():
    value = '{"image": {"count": 2}, "video": {"count": 1}}'
    argv = build({"SM_VLLM_LIMIT_MM_PER_PROMPT": value})
    assert value_of(argv, "--limit-mm-per-prompt") == [value]


# --- values that only look like JSON -------------------------------------------


def test_jinja_chat_template_is_passed_through():
    """A Jinja template starts with '{' but is not JSON; splitting it would
    corrupt the template."""
    value = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    argv = build({"SM_VLLM_CHAT_TEMPLATE": value})
    assert value_of(argv, "--chat-template") == [value]


def test_plain_string_with_spaces_is_one_token():
    argv = build({"SM_VLLM_MODEL": "/opt/ml/model/my model dir"})
    assert value_of(argv, "--model") == ["/opt/ml/model/my model dir"]


def test_malformed_json_array_is_passed_through_untouched():
    """Don't swallow user error: let vLLM report it against the original value."""
    argv = build({"SM_VLLM_LORA_MODULES": '[{"name":"a","path":}]'})
    assert value_of(argv, "--lora-modules") == ['[{"name":"a","path":}]']


def test_multiline_value_is_one_token():
    """`env | grep` split multi-line values across iterations; os.environ does not."""
    value = '{"method": "x",\n "num_speculative_tokens": 1}'
    argv = build({"SM_VLLM_SPECULATIVE_CONFIG": value})
    assert value_of(argv, "--speculative-config") == [value]


# --- unchanged behaviors --------------------------------------------------------


def test_env_key_is_lowercased_and_underscores_become_dashes():
    argv = build({"SM_VLLM_TENSOR_PARALLEL_SIZE": "8"})
    assert value_of(argv, "--tensor-parallel-size") == ["8"]


def test_boolean_true_emits_bare_flag():
    argv = build({"SM_VLLM_ENABLE_LORA": "true"})
    assert "--enable-lora" in argv
    assert value_of(argv, "--enable-lora") == []


def test_boolean_false_omits_flag():
    argv = build({"SM_VLLM_ENABLE_LORA": "false"})
    assert "--enable-lora" not in argv


def test_empty_value_emits_bare_flag():
    argv = build({"SM_VLLM_ENABLE_LORA": ""})
    assert value_of(argv, "--enable-lora") == []


def test_non_prefixed_env_vars_are_ignored():
    argv = build({"PATH": "/usr/bin", "HF_TOKEN": "secret", "SM_VLLM_MODEL": "m"})
    assert "--path" not in argv
    assert "--hf-token" not in argv


def test_port_defaults_to_8080():
    assert value_of(build({}), "--port") == ["8080"]


# --- model resolution ladder ----------------------------------------------------


def test_explicit_sm_vllm_model_wins_over_mounted_dir(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    argv = build(
        {"SM_VLLM_MODEL": "Qwen/Qwen3-8B", "HF_MODEL_ID": "ignored"},
        model_dir=str(tmp_path),
    )
    assert value_of(argv, "--model") == ["Qwen/Qwen3-8B"]
    assert argv.count("--model") == 1


def test_populated_model_dir_is_auto_detected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    argv = build({}, model_dir=str(tmp_path))
    assert value_of(argv, "--model") == [str(tmp_path)]


def test_hf_model_id_used_when_model_dir_empty(tmp_path):
    argv = build({"HF_MODEL_ID": "Qwen/Qwen3-8B"}, model_dir=str(tmp_path))
    assert value_of(argv, "--model") == ["Qwen/Qwen3-8B"]


def test_mounted_dir_takes_precedence_over_hf_model_id(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    argv = build({"HF_MODEL_ID": "Qwen/Qwen3-8B"}, model_dir=str(tmp_path))
    assert value_of(argv, "--model") == [str(tmp_path)]


def test_no_model_source_emits_no_model_flag(tmp_path):
    argv = build({}, model_dir=str(tmp_path))
    assert "--model" not in argv


# --- guards: the shipped entrypoints must actually use the helper ---------------


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda p: p.name)
def test_entrypoint_invokes_the_helper(entrypoint):
    text = entrypoint.read_text()
    assert "sagemaker_args.py" in text, f"{entrypoint.name} no longer calls the helper"
    # NUL-delimited read is what keeps values containing spaces/newlines intact, and
    # `read -d ''` (unlike `mapfile -d`) works on bash < 4.4.
    assert "while IFS= read -r -d '' token" in text
