#!/bin/bash
set -eux
nvidia-smi

cd vllm_source

# Test LoRA adapter loading/unloading via SageMaker endpoints
pytest tests/entrypoints/sagemaker/test_sagemaker_lora_adapters.py -v

# Test stateful session management
pytest tests/entrypoints/sagemaker/test_sagemaker_stateful_sessions.py -v

# Test sagemaker custom middleware
pytest tests/entrypoints/sagemaker/test_sagemaker_middleware_integration.py -v

# Test sagemaker endpoint overrides
pytest tests/entrypoints/sagemaker/test_sagemaker_handler_overrides.py -v

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
python3 offline_inference/prefix_caching.py
python3 offline_inference/llm_engine_example.py
python3 offline_inference/audio_language.py --seed 0
python3 offline_inference/vision_language.py --seed 0
python3 offline_inference/vision_language_multi_image.py --seed 0
python3 others/tensorize_vllm_model.py --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 others/tensorize_vllm_model.py --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
python3 ${BASIC_DIR}/classify.py
python3 ${BASIC_DIR}/embed.py
python3 ${BASIC_DIR}/score.py
python3 offline_inference/spec_decode.py --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048
# https://github.com/vllm-project/vllm/pull/26682 uses slightly more memory in PyTorch 2.9+ causing this test to OOM in 1xL4 GPU
python3 offline_inference/spec_decode.py --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536
