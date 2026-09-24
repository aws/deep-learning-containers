#!/bin/bash
set -eux
nvidia-smi

# Per-test timeout (pytest-timeout) so a stuck readiness loop fails fast, not at 6h.
export PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-600}"

cd vllm_source

SM_TEST_DIR="tests/entrypoints/serve/sagemaker"

# Test LoRA adapter loading/unloading via SageMaker endpoints
pytest ${SM_TEST_DIR}/test_sagemaker_lora_adapters.py -v

# Test stateful session management
pytest ${SM_TEST_DIR}/test_sagemaker_stateful_sessions.py -v

# Test sagemaker custom middleware
pytest ${SM_TEST_DIR}/test_sagemaker_middleware_integration.py -v

# Skipped: test-setup resolves fastapi >=0.137, breaking handler overrides (upstream vLLM #44194).
# pytest ${SM_TEST_DIR}/test_sagemaker_handler_overrides.py -v

# Test LoRA adapter loading/unloading via original OpenAI API server endpoints
pytest tests/entrypoints/serve/lora/test_lora_adapters.py -v

cd examples
pip install tensorizer # for tensorizer test

python3 basic/offline_inference/generate.py --model facebook/opt-125m
python3 basic/offline_inference/chat.py
python3 features/automatic_prefix_caching/prefix_caching_offline.py
python3 generate/multimodal/audio_language_offline.py --seed 0
python3 generate/multimodal/vision_language_offline.py --seed 0
python3 generate/multimodal/vision_language_multi_image_offline.py --seed 0

TENSORIZE="features/tensorize_vllm_model.py"
python3 ${TENSORIZE} --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 ${TENSORIZE} --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors

python3 generate/multimodal/encoder_decoder_multimodal_offline.py --model-type whisper --seed 0
python3 basic/offline_inference/classify.py
python3 basic/offline_inference/embed.py
python3 basic/offline_inference/score.py

SPEC_DECODE="features/speculative_decoding/spec_decode_offline.py"
# vLLM 0.29.0's default CUDA graph memory profiler lowers the effective gpu-memory-utilization,
# and https://github.com/vllm-project/vllm/pull/26682 uses more memory in PyTorch 2.9+, leaving
# too little KV cache on 1xL4. Bump gpu-memory-utilization to reclaim KV headroom.
python3 ${SPEC_DECODE} --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048 --gpu-memory-utilization 0.95
python3 ${SPEC_DECODE} --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536 --gpu-memory-utilization 0.95
