#!/bin/bash
set -eux

nvidia-smi

# Examples Test # 30min
cd vllm_source/examples
pip install tensorizer # for tensorizer test
python3 offline_inference/basic/generate.py --model facebook/opt-125m
# python3 offline_inference/basic/generate.py --model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
python3 offline_inference/basic/chat.py
python3 offline_inference/prefix_caching.py
python3 offline_inference/llm_engine_example.py

# NOTE: Change in Ultravox model changed the class of a audio_processor https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/commit/9a3c571b8fdaf1e66dd3ea61bbcb6db5c70a438e
# vLLM created a fix here https://github.com/vllm-project/vllm/pull/29588 but it is not consumed in vLLM<=0.11
# python3 offline_inference/audio_language.py --seed 0

python3 offline_inference/vision_language.py --seed 0
# broken before v0.12.0: https://github.com/vllm-project/vllm/commit/c64c0b78de4716ef019666663c56b6ceaa019463
# python3 offline_inference/vision_language_pooling.py --seed
# python3 offline_inference/vision_language_multi_image.py --seed 0
python3 others/tensorize_vllm_model.py --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 others/tensorize_vllm_model.py --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
python3 offline_inference/basic/classify.py
python3 offline_inference/basic/embed.py
python3 offline_inference/basic/score.py
VLLM_USE_V1=0 python3 offline_inference/profiling.py --model facebook/opt-125m run_num_steps --num-steps 2
