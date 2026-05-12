#!/bin/bash
# Smoke test for vLLM-Omni SageMaker images
# Uses /invocations with the routing middleware (CustomAttributes: route=<path>)
# Request payload and validation are passed as arguments from the model config.
#
# The <test_request> argument may be either:
#   - a literal request body (JSON or urlencoded form), or
#   - '@/path/to/file' to read the body from a file. Used by the workflow to
#     pass large payloads (e.g. TTS requests with base64-encoded ref_audio)
#     that would exceed shell argument limits.
set -eux

ROUTE="${1:?Usage: $0 <route> <test_request|@file> <validate> [content_type]}"
REQUEST="${2:?Usage: $0 <route> <test_request|@file> <validate> [content_type]}"
VALIDATE="${3:?Usage: $0 <route> <test_request|@file> <validate> [content_type]}"
CONTENT_TYPE="${4:-application/json}"
PORT=8080

if [[ "${REQUEST}" == @* ]]; then
    REQUEST_FILE="${REQUEST#@}"
else
    REQUEST_FILE="$(mktemp)"
    printf '%s' "${REQUEST}" > "${REQUEST_FILE}"
fi

echo "=== vLLM-Omni SageMaker smoke test ==="
echo "Route: ${ROUTE}"
echo "Content-Type: ${CONTENT_TYPE}"
echo "Validate: ${VALIDATE}"

# Wait for server
for i in $(seq 1 300); do
    if curl -s http://localhost:${PORT}/ping >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

curl -sf http://localhost:${PORT}/ping || { echo "Ping failed"; exit 1; }

# Send request via /invocations with route header
if [ "${CONTENT_TYPE}" = "multipart/form-data" ]; then
    CURL_CMD=(curl -sf -X POST "http://localhost:${PORT}/invocations"
      -H "X-Amzn-SageMaker-Custom-Attributes: route=${ROUTE}")
    while IFS= read -r pair; do
        [ -n "${pair}" ] && CURL_CMD+=(-F "${pair}")
    done < <(tr '&' '\n' < "${REQUEST_FILE}")
    CURL_CMD+=(--output /tmp/omni_response --max-time 300)
    "${CURL_CMD[@]}"
else
    curl -sf -X POST http://localhost:${PORT}/invocations \
      -H "Content-Type: application/json" \
      -H "X-Amzn-SageMaker-Custom-Attributes: route=${ROUTE}" \
      -d "@${REQUEST_FILE}" \
      --output /tmp/omni_response --max-time 300
fi

# Validate response
if [[ "${VALIDATE}" == binary_size_gt:* ]]; then
    MIN_SIZE="${VALIDATE#binary_size_gt:}"
    FILE_SIZE=$(stat -c%s /tmp/omni_response 2>/dev/null || stat -f%z /tmp/omni_response)
    echo "Response size: ${FILE_SIZE} bytes (min: ${MIN_SIZE})"
    [ "${FILE_SIZE}" -gt "${MIN_SIZE}" ] || { echo "FAIL: response too small"; exit 1; }

elif [[ "${VALIDATE}" == json_field:* ]]; then
    FIELD="${VALIDATE#json_field:}"
    python3 -c "
import json, sys
data = json.load(open('/tmp/omni_response'))
obj = data
for part in '${FIELD}'.replace(']','').replace('[','.').split('.'):
    if part.isdigit():
        obj = obj[int(part)]
    else:
        obj = obj[part]
assert obj, 'Field ${FIELD} is empty'
print(f'Validated: ${FIELD} present ({type(obj).__name__})')
"
fi

echo "=== vLLM-Omni SageMaker smoke test PASSED ==="
