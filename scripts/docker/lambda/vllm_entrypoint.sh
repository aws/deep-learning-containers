#!/bin/sh
# Serving entrypoint for the Lambda vLLM image.
#
# Kept separate from lambda_entrypoint.sh so serving-engine-specific setup can
# live here without touching the core (base/cupy/pytorch) lambda images. The
# RIE/RIC contract is identical to lambda_entrypoint.sh: when AWS_LAMBDA_RUNTIME_API
# is unset (local testing) we wrap the command in the Runtime Interface Emulator;
# in the real Lambda environment the platform sets it and we exec directly.
#
# The image ENTRYPOINT invokes this as:
#   vllm_entrypoint.sh python -m awslambdaric handler.handler
# so the Runtime Interface Client drives the vLLM offline engine wrapped in
# handler.handler, one invocation per request.

if [ -z "${AWS_LAMBDA_RUNTIME_API}" ]; then
  exec /usr/local/bin/aws-lambda-rie "$@"
else
  exec "$@"
fi
