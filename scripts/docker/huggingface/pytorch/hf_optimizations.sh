#!/usr/bin/env bash
#
# HuggingFace PyTorch training auto-optimization layer.
#
# SOURCED (not executed) by the entrypoint before the training process starts.
# Applies sensible throughput/stability defaults so HF training runs well out of
# the box without hand-tuning. Two rules:
#   1. Nothing the user already set is overridden — explicit env always wins.
#   2. Every feature has an HF_ENABLE_* toggle; HF_ENABLE_OPTIMIZATIONS=0 kills
#      the whole layer.

: "${HF_ENABLE_OPTIMIZATIONS:=1}"
if [[ "${HF_ENABLE_OPTIMIZATIONS}" != "1" ]]; then
  echo "[hf-opt] disabled (HF_ENABLE_OPTIMIZATIONS=${HF_ENABLE_OPTIMIZATIONS})"
  return 0 2>/dev/null || exit 0
fi

# ---- feature toggles -------------------------------------------------------
: "${HF_ENABLE_EXPANDABLE_SEGMENTS:=1}"
: "${HF_ENABLE_HF_TRANSFER:=1}"
: "${HF_ENABLE_TOKENIZERS_GUARD:=1}"
: "${HF_ENABLE_OMP_DEFAULT:=1}"
: "${HF_ENABLE_CUDA_DEVICE_MAX_CONNECTIONS:=0}"  # opt-in

hf_opt_log() { echo "[hf-opt] $*"; }

# ---- 1. CUDA caching allocator: cut fragmentation OOM on long runs ----------
# expandable_segments (the CUDA VMM allocator) reduces fragmentation that
# otherwise causes spurious OOMs as sequence lengths / batch sizes vary across
# training steps. Safe default for training (no KV connectors to conflict with,
# unlike the inference image).
if [[ "${HF_ENABLE_EXPANDABLE_SEGMENTS}" == "1" && -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  hf_opt_log "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
fi

# ---- 2. Fast Hub transfers (hf-transfer Rust downloader) --------------------
if [[ "${HF_ENABLE_HF_TRANSFER}" == "1" && -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]]; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  hf_opt_log "HF_HUB_ENABLE_HF_TRANSFER=1"
fi

# ---- 3. tokenizers fork guard ----------------------------------------------
# Silences the fast-tokenizers fork warning and avoids the rare DataLoader
# worker deadlock when tokenizers' Rust threadpool is forked.
if [[ "${HF_ENABLE_TOKENIZERS_GUARD}" == "1" && -z "${TOKENIZERS_PARALLELISM:-}" ]]; then
  export TOKENIZERS_PARALLELISM=false
  hf_opt_log "TOKENIZERS_PARALLELISM=false"
fi

# ---- 4. OMP threads --------------------------------------------------------
# torch otherwise grabs all cores per process; with many ranks per node that
# oversubscribes the CPU and slows the data pipeline. Default to 1 per rank.
if [[ "${HF_ENABLE_OMP_DEFAULT}" == "1" && -z "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS=1
  hf_opt_log "OMP_NUM_THREADS=1"
fi

# ---- 5. CUDA_DEVICE_MAX_CONNECTIONS (opt-in) -------------------------------
# Recommended for tensor-parallel / Transformer Engine compute-comm overlap, but
# can hurt other workloads, so it's off by default.
if [[ "${HF_ENABLE_CUDA_DEVICE_MAX_CONNECTIONS}" == "1" && -z "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  hf_opt_log "CUDA_DEVICE_MAX_CONNECTIONS=1"
fi
