"""Translate SM_VLLM_* environment variables into vLLM server CLI arguments.

SageMaker passes configuration as environment variables, so the entrypoint has to
turn ``SM_VLLM_TENSOR_PARALLEL_SIZE=8`` into ``--tensor-parallel-size 8``. Doing that
in shell used to collapse every value into a single argv token, which made flags
declared ``nargs="+"`` impossible to use: ``--lora-modules`` (and ~20 other
list-typed config fields such as --served-model-name and --custom-ops) need one argv
token per value, and a single token like ``[{...},{...}]`` fails inside vLLM's
``LoRAParserAction``.

The translation mirrors what vLLM already does for ``--config file.yaml`` in
``vllm/utils/argparse_utils.py``:

  * JSON array  -> the flag plus one argv token per element
  * JSON object -> the flag plus a single argv token (the object)
  * anything else -> the flag plus the raw value as a single token

Because the decision is made from the *value's* syntax rather than a per-flag table,
no list of flags has to be maintained as vLLM adds arguments. The exception is the
handful of flags where vLLM deliberately wants an array as one token; those are
listed in SINGLE_TOKEN_FLAGS below.

Tokens are written to stdout NUL-delimited so the shell can read them back into an
array without word-splitting values that contain spaces or newlines. Informational
messages go to stderr to keep stdout parseable.
"""

import json
import os
import sys
from typing import Any, Iterable, List, Mapping, Optional

PREFIX = "SM_VLLM_"
ARG_PREFIX = "--"
DEFAULT_PORT = "8080"
MODEL_DIR = "/opt/ml/model"

# Flags whose value stays a single argv token even when it is a JSON array.
# vLLM strips nargs from these on purpose (see FrontendArgs._customize_cli_kwargs in
# vllm/entrypoints/openai/cli_args.py): the first three are typed ``json.loads`` so the
# array *is* the value, and --middleware is ``action="append"``, taking one value per
# occurrence.
SINGLE_TOKEN_FLAGS = frozenset(
    {
        "--allowed-origins",
        "--allowed-methods",
        "--allowed-headers",
        "--middleware",
    }
)


def flag_for(env_key: str) -> str:
    """``SM_VLLM_TENSOR_PARALLEL_SIZE`` -> ``--tensor-parallel-size``."""
    name = env_key[len(PREFIX) :].lower().replace("_", "-")
    return f"{ARG_PREFIX}{name}"


def as_token(element: Any) -> str:
    """Render one element of a JSON array as a single argv token.

    Nested objects/arrays are re-serialized compactly so they survive as JSON (vLLM
    parses --lora-modules elements with ``json.loads``); scalars become plain strings,
    with booleans lowercased to the spelling vLLM's parsers accept.
    """
    if isinstance(element, (dict, list)):
        return json.dumps(element, separators=(",", ":"))
    if isinstance(element, bool):
        return "true" if element else "false"
    if element is None:
        return ""
    return str(element)


def json_object_sequence(value: str) -> Optional[List[dict]]:
    """Parse ``{...} {...}`` (the spelling vLLM's own CLI takes) into its objects.

    Returns None if the string is not a whitespace-separated run of JSON objects, so
    callers can fall back to passing the value through untouched. A Jinja template
    such as ``{% for m in messages %}`` lands here and is correctly rejected.
    """
    decoder = json.JSONDecoder()
    objects: List[dict] = []
    index = 0
    length = len(value)
    while index < length:
        try:
            parsed, end = decoder.raw_decode(value, index)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        objects.append(parsed)
        index = end
        while index < length and value[index].isspace():
            index += 1
    return objects or None


def tokens_for(flag: str, value: str) -> Optional[List[str]]:
    """Return the argv tokens that follow `flag`.

    An empty list means "emit the flag with no values" (a boolean-style flag); None
    means "omit the flag entirely", which is what an empty JSON array asks for since
    argparse rejects a nargs='+' flag with zero values.
    """
    if not value:
        return []

    stripped = value.strip()

    if flag in SINGLE_TOKEN_FLAGS:
        return [value]

    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return [value]  # not valid JSON: let vLLM report the original value
        if isinstance(parsed, list):
            if not parsed:
                return None
            return [as_token(element) for element in parsed]
        return [value]

    if stripped.startswith("{") and stripped.endswith("}"):
        objects = json_object_sequence(stripped)
        if objects is not None and len(objects) > 1:
            return [json.dumps(obj, separators=(",", ":")) for obj in objects]
        # A single JSON object is passed through verbatim rather than re-serialized so
        # its exact spelling reaches vLLM.
        return [value]

    return [value]


def resolve_model(env: Mapping[str, str], model_dir: str) -> List[str]:
    """Pick the model source when SM_VLLM_MODEL is not set.

    Precedence: an explicit SM_VLLM_MODEL (handled by the generic loop, so nothing is
    added here), then a populated model dir, then HF_MODEL_ID.
    """
    if env.get(f"{PREFIX}MODEL"):
        return []

    if os.path.isdir(model_dir) and os.listdir(model_dir):
        log(f"INFO: {PREFIX}MODEL not set, auto-detected model at {model_dir}")
        return ["--model", model_dir]

    if env.get("HF_MODEL_ID"):
        log(f"INFO: {PREFIX}MODEL not set, using HF_MODEL_ID={env['HF_MODEL_ID']}")
        return ["--model", env["HF_MODEL_ID"]]

    log(
        f"WARNING: No model specified. Set {PREFIX}MODEL, HF_MODEL_ID, "
        f"or mount a model to {model_dir}."
    )
    return []


def log(message: str) -> None:
    print(message, file=sys.stderr)


def build_args(env: Mapping[str, str], model_dir: str = MODEL_DIR) -> List[str]:
    """Build the full argv list (minus the program itself) for the vLLM server."""
    args: List[str] = ["--port", DEFAULT_PORT]
    args += resolve_model(env, model_dir)

    for key in sorted(env):
        if not key.startswith(PREFIX):
            continue
        flag = flag_for(key)
        value = env[key]
        lowered = value.strip().lower()

        # Boolean flags: true -> bare flag, false -> omitted entirely.
        if lowered == "true":
            args.append(flag)
            continue
        if lowered == "false":
            continue

        tokens = tokens_for(flag, value)
        if tokens is None:
            log(f"WARNING: {key} is an empty list; skipping {flag}.")
            continue
        args.append(flag)
        args += tokens

    return args


def emit(tokens: Iterable[str]) -> None:
    """Write tokens NUL-delimited for `mapfile -t -d ''` on the shell side."""
    sys.stdout.write("".join(f"{token}\0" for token in tokens))


def main() -> None:
    args = build_args(os.environ)
    log(f"INFO: vLLM server arguments: {args}")
    emit(args)


if __name__ == "__main__":
    main()
