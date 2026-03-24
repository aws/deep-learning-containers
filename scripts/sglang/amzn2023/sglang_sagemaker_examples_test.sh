#!/bin/bash
set -eux

# Run the same examples as EC2 — SGLang does not have SageMaker-specific upstream tests yet.
# SageMaker-specific validation is handled by the endpoint-test job (test/sglang/sagemaker/test_sm_endpoint.py).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/sglang_ec2_examples_test.sh"
