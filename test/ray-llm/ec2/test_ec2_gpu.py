"""EC2 single-GPU integration test for the ray-llm DLC.

Boots the container locally via `docker run --gpus all`, hits the OpenAI-
compatible endpoints, and validates response schema.
"""

import logging

import requests

REQUEST_TIMEOUT = 120

LOGGER = logging.getLogger(__name__)

MODEL_ID = "ministral"


def _post(port, path, payload):
    resp = requests.post(f"http://localhost:{port}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _validate(body, label):
    for field in ("id", "object", "choices", "usage"):
        assert field in body, f"{label}: response missing '{field}'"
    tokens = body.get("usage", {}).get("completion_tokens", 0)
    assert tokens > 0, f"{label}: completion_tokens={tokens}"


def test_models_lists_ministral(container):
    resp = requests.get(f"http://localhost:{container['port']}/v1/models", timeout=10)
    resp.raise_for_status()
    body = resp.json()
    ids = [m["id"] for m in body.get("data", [])]
    assert MODEL_ID in ids, f"/v1/models missing {MODEL_ID}: {ids}"


def test_completions(container):
    body = _post(
        container["port"],
        "/v1/completions",
        {"model": MODEL_ID, "prompt": "Hello, how are you?", "max_tokens": 100, "temperature": 0.7},
    )
    LOGGER.info(f"/v1/completions response: {body}")
    _validate(body, "/v1/completions")


def test_chat_completions(container):
    body = _post(
        container["port"],
        "/v1/chat/completions",
        {
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": "What are the benefits of using FSx Lustre with EKS?"}
            ],
            "max_tokens": 200,
            "temperature": 0.7,
        },
    )
    LOGGER.info(f"/v1/chat/completions response: {body}")
    _validate(body, "/v1/chat/completions")
