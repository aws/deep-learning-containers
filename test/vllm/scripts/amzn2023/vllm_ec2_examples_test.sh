#!/bin/bash
set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run shared vLLM EC2 example tests
bash "${SCRIPT_DIR}/../vllm_ec2_examples_test.sh"

# Run amzn2023-specific tests
echo "=== Running LMCache bidirectional feature tests ==="
python3 "${SCRIPT_DIR}/test_lmcache_bidirectional.py"
