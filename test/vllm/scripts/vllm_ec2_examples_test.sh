#!/bin/bash
set -eux

nvidia-smi
cd vllm_source/examples
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
python3 ${SPEC_DECODE} --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048
# https://github.com/vllm-project/vllm/pull/26682 uses slightly more memory in PyTorch 2.9+ causing this test to OOM in 1xL4 GPU
python3 ${SPEC_DECODE} --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536
