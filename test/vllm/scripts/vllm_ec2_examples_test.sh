#!/bin/bash
set -eux

nvidia-smi
cd vllm_source/examples
pip install tensorizer # for tensorizer test

# vLLM ≥0.29 reorganised the examples tree; detect which layout we have.
if [ -d "offline_inference" ]; then
  # New layout (v0.29+)
  python3 offline_inference/basic/generate.py --model facebook/opt-125m
  python3 offline_inference/basic/chat.py
  python3 offline_inference/basic/classify.py
  python3 offline_inference/basic/embed.py
  python3 offline_inference/basic/score.py

  # automatic_prefix_caching may live under features/ in v0.29+
  APC=$(find . -path '*/automatic_prefix_caching*' -name '*.py' | head -1)
  [ -n "$APC" ] && python3 "$APC" || true

  python3 offline_inference/multimodal/audio_language.py --seed 0
  python3 offline_inference/multimodal/vision_language.py --seed 0
  python3 offline_inference/multimodal/vision_language_multi_image.py --seed 0
  # encoder_decoder_multimodal may have been renamed in v0.29+
  ECDM=$(find . -path '*/encoder_decoder*' -name '*.py' | head -1)
  [ -n "$ECDM" ] && python3 "$ECDM" --model-type whisper --seed 0 || true

  # tensorize script may live under features/ in v0.29+
  TENSORIZE=$(find . -path '*tensorize_vllm_model.py' | head -1)
  [ -n "$TENSORIZE" ] && python3 ${TENSORIZE} --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 ${TENSORIZE} --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors

  # spec_decode may live under features/ in v0.29+
  SPEC_DECODE=$(find . -path '*spec_decode*.py' -not -path '*/test*' | head -1)
  [ -z "$SPEC_DECODE" ] && SPEC_DECODE="offline_inference/spec_decode.py"
  python3 ${SPEC_DECODE} --test --method eagle --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 2048
  python3 ${SPEC_DECODE} --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536
else
  # Old layout (pre-0.29)
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
  python3 ${SPEC_DECODE} --test --method eagle3 --num_spec_tokens 3 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --temp 0 --top-p 1.0 --top-k -1 --tp 1 --enable-chunked-prefill --max-model-len 1536
fi
