#!/bin/bash
set -eux
nvidia-smi

# Per-test timeout (pytest-timeout) so a stuck readiness loop fails fast, not at 6h.
export PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-600}"

cd vllm_source

# vLLM v0.23.0 moved the sagemaker entrypoint tests under serve/
if [ -d "tests/entrypoints/serve/sagemaker" ]; then
  SM_TEST_DIR="tests/entrypoints/serve/sagemaker"
else
  SM_TEST_DIR="tests/entrypoints/sagemaker"
fi

# Test LoRA adapter loading/unloading via SageMaker endpoints
pytest ${SM_TEST_DIR}/test_sagemaker_lora_adapters.py -v

# Test stateful session management
pytest ${SM_TEST_DIR}/test_sagemaker_stateful_sessions.py -v

# Test sagemaker custom middleware
pytest ${SM_TEST_DIR}/test_sagemaker_middleware_integration.py -v

# Test sagemaker endpoint overrides
pytest ${SM_TEST_DIR}/test_sagemaker_handler_overrides.py -v

# Test LoRA adapter loading/unloading via original OpenAI API server endpoints
pytest tests/entrypoints/serve/lora/test_lora_adapters.py -v

cd examples
pip install tensorizer # for tensorizer test

# vLLM 0.18.0 moved basic scripts from offline_inference/basic/ to basic/offline_inference/
if [ -d "basic/offline_inference" ]; then
	BASIC_DIR="basic/offline_inference"
else
	BASIC_DIR="offline_inference/basic"
fi

python3 ${BASIC_DIR}/generate.py --model facebook/opt-125m
python3 ${BASIC_DIR}/chat.py
if [ -f "features/automatic_prefix_caching/prefix_caching_offline.py" ]; then
  python3 features/automatic_prefix_caching/prefix_caching_offline.py
else
  python3 offline_inference/prefix_caching.py
fi
# vLLM v0.20.1rc0 moved multimodal examples to generate/multimodal/
if [ -d "generate/multimodal" ]; then
  MM_DIR="generate/multimodal"
  python3 ${MM_DIR}/audio_language_offline.py --seed 0
  python3 ${MM_DIR}/vision_language_offline.py --seed 0
  python3 ${MM_DIR}/vision_language_multi_image_offline.py --seed 0
else
  python3 offline_inference/audio_language.py --seed 0
  python3 offline_inference/vision_language.py --seed 0
  python3 offline_inference/vision_language_multi_image.py --seed 0
fi
# vLLM post-v0.20.2 moved tensorize example to features/
if [ -f "features/tensorize_vllm_model.py" ]; then
  TENSORIZE="features/tensorize_vllm_model.py"
else
  TENSORIZE="others/tensorize_vllm_model.py"
fi
python3 ${TENSORIZE} --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 ${TENSORIZE} --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
if [ -d "generate/multimodal" ]; then
  python3 generate/multimodal/encoder_decoder_multimodal_offline.py --model-type whisper --seed 0
else
  python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
fi
python3 ${BASIC_DIR}/classify.py
python3 ${BASIC_DIR}/embed.py
python3 ${BASIC_DIR}/score.py
if [ -f "features/speculative_decoding/spec_decode_offline.py" ]; then
  SPEC_DECODE="features/speculative_decoding/spec_decode_offline.py"
else
  SPEC_DECODE="offline_inference/spec_decode.py"
fi
python3 ${SPEC_DECODE} --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048
# https://github.com/vllm-project/vllm/pull/26682 uses slightly more memory in PyTorch 2.9+ causing this test to OOM in 1xL4 GPU
python3 ${SPEC_DECODE} --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536
