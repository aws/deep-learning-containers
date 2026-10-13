"""Shared helpers for the config-driven vLLM SageMaker serving tests.

Used by test_sm_local_serving.py (``fleet:`` entries) and test_sm_model_serving.py
(``instance_type:`` entries); see the header of
.github/config/model-tests/vllm-sagemaker-endpoint-tests.yml.

CLI, for ``docker run --env-file``:
    python3 test/vllm/sagemaker/amzn2023/sm_serving_common.py --model-name <name>
"""

import argparse
import io
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[4]
CONFIG_PATH = REPO_ROOT / ".github/config/model-tests/vllm-sagemaker-endpoint-tests.yml"


def load_sagemaker_config(model_name=None):
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("sagemaker", [])
    s3_prefix = cfg.get("s3_prefix", "")
    if model_name:
        models = [m for m in models if m["name"] == model_name]
    for m in models:
        if "s3_model" in m:
            m["s3_path"] = f"{s3_prefix}/{m['s3_model']}"
    return models


def container_env(model_cfg):
    """Container environment: the entry's ``env`` plus its flattened chat template."""
    env_vars = dict(model_cfg.get("env", {}))
    not_strings = [key for key, value in env_vars.items() if not isinstance(value, str)]
    if not_strings:
        raise ValueError(
            f"{model_cfg['name']}: env values must be strings; quote {', '.join(not_strings)}"
        )
    chat_template_file = model_cfg.get("chat_template_file")
    if chat_template_file:
        template_path = REPO_ROOT / chat_template_file
        env_vars["SM_VLLM_CHAT_TEMPLATE"] = flatten_jinja(template_path.read_text())
    return env_vars


def _download_s3(s3_client, s3_uri):
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    buf = io.BytesIO()
    s3_client.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return buf.read()


def build_multipart_body(request_cfg, s3_client):
    boundary = "----FormBoundary7MA4YWxkTrZu0gW"
    parts = []

    for field_name, field_value in request_cfg.get("fields", {}).items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'
            f"{field_value}\r\n"
        )

    if "file_s3" in request_cfg:
        file_data = _download_s3(s3_client, request_cfg["file_s3"])
        file_field = request_cfg.get("file_field", "file")
        file_name = request_cfg.get("file_name", "input.bin")
        file_ct = request_cfg.get("file_content_type", "application/octet-stream")

        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'
            f"Content-Type: {file_ct}\r\n\r\n"
        )
        body = "".join(parts).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()
    else:
        parts.append(f"--{boundary}--\r\n")
        body = "".join(parts).encode()

    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def build_json_body(request_cfg):
    payload = request_cfg.get("body", {})
    return json.dumps(payload).encode(), "application/json"


def validate_response(response_body, raw_bytes, rule):
    if rule.startswith("contains:"):
        expected = rule[len("contains:") :]
        if isinstance(response_body, dict):
            text = response_body.get("text", "")
            if not text:
                text = json.dumps(response_body)
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
        assert expected.lower() in text.lower(), (
            f"Expected '{expected}' in response, got: {text[:500]}"
        )

    elif rule.startswith("json_field:"):
        field_path = rule[len("json_field:") :]
        snippet = raw_bytes[:200].decode("utf-8", errors="replace")
        assert response_body is not None, (
            f"Response is not JSON, so field '{field_path}' cannot be checked: {snippet}"
        )
        obj = response_body
        try:
            for part in field_path.replace("]", "").replace("[", ".").split("."):
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = obj[part]
        except (KeyError, IndexError, TypeError):
            raise AssertionError(f"Field '{field_path}' not found in response: {snippet}") from None
        assert obj, f"Field '{field_path}' is empty in response"

    elif rule.startswith("binary_size_gt:"):
        min_size = int(rule[len("binary_size_gt:") :])
        assert len(raw_bytes) > min_size, f"Response size {len(raw_bytes)} <= {min_size}"

    else:
        raise ValueError(f"Unknown validation rule: {rule}")


def flatten_jinja(template_str):
    """Flatten newlines into ``{{ "\\n" }}`` so the template is a single-line env value.

    Line-oriented env transports, such as a ``docker run --env-file`` line, would
    otherwise split a multi-line value. vLLM's ``--chat-template`` falls back to
    inline Jinja when the value isn't a valid file path and contains ``{``, ``}``,
    or newline, and each ``{{ "\\n" }}`` expression evaluates to a real newline at
    request time.

    No literal-quote wrapping is applied. ``standard-supervisor`` (>=0.1.15,
    pinned in docker/vllm/Dockerfile.amzn2023) ``shlex.quote()``s each argv
    element before the supervisord ``shlex.split`` round-trip, so the inner
    double quotes survive intact. Adding outer single quotes here would pass
    them through verbatim into the rendered template.
    """
    return template_str.replace("\n", '{{ "\\n" }}')


def main():
    parser = argparse.ArgumentParser(
        description="Print a config entry's container env as docker --env-file lines."
    )
    parser.add_argument("--model-name", required=True, help="Config entry name")
    args = parser.parse_args()
    models = load_sagemaker_config(args.model_name)
    if len(models) != 1:
        sys.exit(f"{args.model_name!r} matches {len(models)} entries in {CONFIG_PATH.name}, not 1")
    for key, value in container_env(models[0]).items():
        if "\n" in value or "\r" in value:
            sys.exit(f"{key} contains a line break, which an --env-file line cannot carry")
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
