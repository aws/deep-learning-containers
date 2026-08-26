#!/bin/bash
set -euo pipefail

# SGLang Tool-Calling Smoke Test — parser selected by caller via extra_args
# (e.g. --tool-call-parser qwen25); required/named use the default xgrammar backend.
# Usage: sglang_tool_calling_smoke_test.sh <model_dir> <model_name> [extra_args...]

MODEL_DIR="${1:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
MODEL_NAME="${2:?Usage: $0 <model_dir> <model_name> [extra_args...]}"
shift 2
EXTRA_ARGS="$*"

SGLANG_PORT="${SGLANG_PORT:-30000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
HEALTH_INTERVAL=10

echo "=== Tool-Calling Smoke Test: ${MODEL_NAME} ==="
echo "=== Model directory: ${MODEL_DIR} ==="
echo "=== Extra args: ${EXTRA_ARGS} ==="

echo "=== Starting SGLang server ==="
# shellcheck disable=SC2086
python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --host 0.0.0.0 \
  --port "${SGLANG_PORT}" \
  ${EXTRA_ARGS} &
SGLANG_PID=$!

cleanup() {
  kill "${SGLANG_PID}" 2>/dev/null || true
  wait "${SGLANG_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Waiting for health check ==="
elapsed=0
while [ "${elapsed}" -lt "${HEALTH_TIMEOUT}" ]; do
  if curl -sf http://localhost:${SGLANG_PORT}/health >/dev/null 2>&1; then
    echo "Server healthy after ${elapsed}s"
    break
  fi
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "ERROR: SGLang process died"; exit 1
  fi
  sleep "${HEALTH_INTERVAL}"
  elapsed=$((elapsed + HEALTH_INTERVAL))
done
[ "${elapsed}" -ge "${HEALTH_TIMEOUT}" ] && echo "ERROR: timeout" && exit 1

echo "=== Running tool-calling tests ==="
python3 - "${MODEL_DIR}" "${SGLANG_PORT}" "${MODEL_NAME}" << 'PYEOF'
import httpx, json, sys

model_dir, port, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = f"http://localhost:{port}"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def chat(messages, tools=None, tool_choice=None, max_tokens=256):
    payload = {"model": model_dir, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0}
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    resp = httpx.post(f"{BASE}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def assert_tool_call(resp, expected_name):
    """Assert the response carries a single well-formed tool call."""
    choice = resp["choices"][0]
    msg = choice["message"]
    calls = msg.get("tool_calls")
    assert calls, f"No tool_calls in response: {json.dumps(msg)[:400]}"
    fn = calls[0]["function"]
    assert fn["name"] == expected_name, f"Expected {expected_name}, got {fn['name']}"
    # arguments is a JSON-encoded string per the OpenAI schema.
    args = json.loads(fn["arguments"])
    assert isinstance(args, dict), f"arguments not a JSON object: {fn['arguments']}"
    # finish_reason=tool_calls is OpenAI-standard but not contractually documented; warn only.
    if choice.get("finish_reason") != "tool_calls":
        print(f"  WARN: finish_reason={choice.get('finish_reason')} (expected tool_calls)")
    return fn, args


# --- Test 1: tool_choice=auto (observational, warn-only: no grammar enforcement) ---
print("\n--- Test 1: auto tool_choice (observational) ---")
r = chat(
    [{"role": "user", "content": "What is the weather in Paris right now?"}],
    tools=[WEATHER_TOOL], tool_choice="auto",
)
auto_msg = r["choices"][0]["message"]
if auto_msg.get("tool_calls"):
    fn = auto_msg["tool_calls"][0]["function"]
    print(f"  OK: auto -> {fn['name']}({fn['arguments']})")
else:
    print(f"  WARN: no tool call in auto mode (non-deterministic): "
          f"{(auto_msg.get('content') or '')[:120]}")

# --- Test 2: tool_choice=required forces a call even for a vague prompt ---
print("\n--- Test 2: required tool_choice ---")
r = chat(
    [{"role": "user", "content": "Tell me about the weather."}],
    tools=[WEATHER_TOOL], tool_choice="required",
)
assert_tool_call(r, "get_current_weather")
print("  PASS: required forced a tool call")

# --- Test 3: named tool_choice selects the requested function ---
print("\n--- Test 3: named-function tool_choice ---")
r = chat(
    [{"role": "user", "content": "Weather in Tokyo?"}],
    tools=[WEATHER_TOOL],
    tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
)
assert_tool_call(r, "get_current_weather")
print("  PASS: named tool_choice honored")

# --- Test 4: multi-turn — feed the tool result back and get a final answer ---
print("\n--- Test 4: multi-turn tool result ---")
first = chat(
    [{"role": "user", "content": "What is the weather in Paris right now?"}],
    tools=[WEATHER_TOOL], tool_choice="required",
)
choice = first["choices"][0]
call = choice["message"]["tool_calls"][0]
followup = chat(
    [
        {"role": "user", "content": "What is the weather in Paris right now?"},
        {"role": "assistant", "content": choice["message"].get("content") or "",
         "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
         "content": json.dumps({"city": "Paris", "temperature": 18, "unit": "celsius"})},
    ],
    # tool_choice=none forces a text answer instead of another tool call.
    tools=[WEATHER_TOOL], tool_choice="none",
)
answer = followup["choices"][0]["message"].get("content") or ""
assert answer.strip(), f"Empty final answer after tool result: {json.dumps(followup)[:400]}"
print(f"  PASS: multi-turn final answer ({len(answer)} chars)")

# --- Test 5: no tools -> normal completion, no tool_calls ---
print("\n--- Test 5: no-tools completion still works ---")
r = chat([{"role": "user", "content": "Say hello in one word."}])
msg = r["choices"][0]["message"]
assert not msg.get("tool_calls"), "Unexpected tool_calls when no tools were provided"
assert (msg.get("content") or "").strip(), "Empty content for plain completion"
print("  PASS: plain completion unaffected")

print(f"\n=== All tool-calling tests passed for {model_name} ===")
PYEOF

echo "=== PASSED: ${MODEL_NAME} tool-calling smoke test ==="
