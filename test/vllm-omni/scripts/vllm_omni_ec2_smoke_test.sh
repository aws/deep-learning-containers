#!/bin/bash
# Smoke test for vLLM-Omni EC2 images
# Uses the OpenAI-compatible API directly (no /invocations middleware).
# Request payload and validation are passed as arguments from the model config.
set -eux

ROUTE="${1:?Usage: $0 <route> <test_request> <validate> [content_type]}"
REQUEST="${2:?Usage: $0 <route> <test_request> <validate> [content_type]}"
VALIDATE="${3:?Usage: $0 <route> <test_request> <validate> [content_type]}"
CONTENT_TYPE="${4:-application/json}"
PORT=8080

echo "=== vLLM-Omni EC2 smoke test ==="
echo "Route: ${ROUTE}"
echo "Content-Type: ${CONTENT_TYPE}"
echo "Validate: ${VALIDATE}"

# Wait for server
for i in $(seq 1 300); do
    if curl -s http://localhost:${PORT}/health >/dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

curl -sf http://localhost:${PORT}/health || { echo "Health check failed"; exit 1; }

# Send request directly to the API endpoint
if [ "${CONTENT_TYPE}" = "multipart/form-data" ]; then
    # Convert key=value&key2=value2 to -F flags
    CURL_ARGS=""
    IFS='&' read -ra PAIRS <<< "${REQUEST}"
    for pair in "${PAIRS[@]}"; do
        CURL_ARGS="${CURL_ARGS} -F ${pair}"
    done
    eval curl -sf -X POST "http://localhost:${PORT}${ROUTE}" ${CURL_ARGS} --output /tmp/omni_response --max-time 300
else
    curl -sf -X POST "http://localhost:${PORT}${ROUTE}" \
      -H "Content-Type: application/json" \
      -d "${REQUEST}" \
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

echo "=== vLLM-Omni EC2 smoke test PASSED ==="
