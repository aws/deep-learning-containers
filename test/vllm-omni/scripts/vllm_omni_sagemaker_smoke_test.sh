#!/bin/bash
# Smoke test for vLLM-Omni SageMaker images
# Uses /invocations with the routing middleware (CustomAttributes: route=<path>)
# Request payload and validation are passed as arguments from the model config.
set -eux

ROUTE="${1:?Usage: $0 <route> <test_request_json> <validate>}"
REQUEST="${2:?Usage: $0 <route> <test_request_json> <validate>}"
VALIDATE="${3:?Usage: $0 <route> <test_request_json> <validate>}"
PORT=8080

echo "=== vLLM-Omni SageMaker smoke test ==="
echo "Route: ${ROUTE}"
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
curl -sf -X POST http://localhost:${PORT}/invocations \
  -H "Content-Type: application/json" \
  -H "X-Amzn-SageMaker-Custom-Attributes: route=${ROUTE}" \
  -d "${REQUEST}" \
  --output /tmp/omni_response --max-time 300

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
# Navigate nested field like data[0].b64_json
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
