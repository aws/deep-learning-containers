#!/bin/bash
set -eux

nvidia-smi

cd sglang_source

# Offline batch inference (analog to vLLM's offline_inference/basic/generate.py)
python3 examples/runtime/engine/offline_batch_inference.py --model facebook/opt-125m

# Correctness test: validates prefill/decode logits without a server
python3 -m sglang.bench_one_batch --model-path facebook/opt-125m --correct

# Offline throughput benchmark
python3 -m sglang.bench_offline_throughput --model facebook/opt-125m --num-prompts 10
