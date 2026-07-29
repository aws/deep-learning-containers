#!/usr/bin/env bash
#
# HuggingFace vLLM auto-optimization layer.
#
# SOURCED (not executed) by the entrypoint before the server starts. Two rules:
#   1. Nothing the user already set is overridden — explicit env always wins.
#   2. Every feature has an HF_ENABLE_* toggle; HF_ENABLE_OPTIMIZATIONS=0 kills
#      the whole layer.

# The entrypoint consumes these even when the layer is disabled.
export HF_LMCACHE_KV_CONFIG=""
: "${HF_ENABLE_RUNAI_STREAMER:=0}" # opt-in
export HF_ENABLE_RUNAI_STREAMER

: "${HF_ENABLE_OPTIMIZATIONS:=1}"
if [[ "${HF_ENABLE_OPTIMIZATIONS}" != "1" ]]; then
  echo "[hf-opt] disabled (HF_ENABLE_OPTIMIZATIONS=${HF_ENABLE_OPTIMIZATIONS})"
  return 0 2>/dev/null || exit 0
fi

# ---- feature toggles -------------------------------------------------------
: "${HF_ENABLE_EXPANDABLE_SEGMENTS:=1}"
: "${HF_ENABLE_LMCACHE:=1}"
: "${HF_LMCACHE_CPU_FRACTION:=0.5}" # share of total RAM for the LMCache CPU pool
: "${HF_ENABLE_XET_HIGH_PERFORMANCE:=1}"
: "${HF_ENABLE_TOKENIZERS_GUARD:=1}"

hf_opt_log() { echo "[hf-opt] $*"; }

# ---- 1. CUDA caching allocator: reduce fragmentation / OOM ------------------
# expandable_segments (the CUDA VMM allocator) is INCOMPATIBLE with KV connectors
# that pin/register KV memory: the allocator can remap KV virtual addresses to
# different physical pages and invalidate those registrations, so vLLM rejects
# the combination. LMCache (our default KV connector) therefore takes precedence
# and we only set the allocator when LMCache is off.
if [[ "${HF_ENABLE_EXPANDABLE_SEGMENTS}" == "1" && -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  if [[ "${HF_ENABLE_LMCACHE}" == "1" ]]; then
    hf_opt_log "expandable_segments skipped (incompatible with the LMCache KV connector); LMCache takes precedence"
  else
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    hf_opt_log "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  fi
fi

# ---- 2. LMCache CPU KV-offload ---------------------------------------------
if [[ "${HF_ENABLE_LMCACHE}" == "1" ]]; then
  _total_gb="$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null)"
  : "${_total_gb:=0}"
  _cpu_gb="$(awk "BEGIN{v=${_total_gb}*${HF_LMCACHE_CPU_FRACTION}; if(v<5)v=5; printf \"%d\", v}")"
  : "${LMCACHE_LOCAL_CPU:=True}"
  export LMCACHE_LOCAL_CPU
  : "${LMCACHE_MAX_LOCAL_CPU_SIZE:=${_cpu_gb}}"
  export LMCACHE_MAX_LOCAL_CPU_SIZE
  : "${LMCACHE_CHUNK_SIZE:=256}"
  export LMCACHE_CHUNK_SIZE
  HF_LMCACHE_KV_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
  hf_opt_log "LMCache CPU offload enabled (LMCACHE_MAX_LOCAL_CPU_SIZE=${LMCACHE_MAX_LOCAL_CPU_SIZE} GB, chunk=${LMCACHE_CHUNK_SIZE})"
  unset _total_gb _cpu_gb
fi
export HF_LMCACHE_KV_CONFIG

# ---- 3. Fast Hub weight downloads (hf-xet high-performance mode) ------------
# Saturates network bandwidth and CPU cores for parallel chunk downloads — cuts
# model pull time on endpoint cold starts / scale-out. Successor to hf_transfer
# (huggingface_hub >= 1.0 removed HF_HUB_ENABLE_HF_TRANSFER).
if [[ "${HF_ENABLE_XET_HIGH_PERFORMANCE}" == "1" && -z "${HF_XET_HIGH_PERFORMANCE:-}" ]]; then
  export HF_XET_HIGH_PERFORMANCE=1
  hf_opt_log "HF_XET_HIGH_PERFORMANCE=1"
fi

# ---- 4. tokenizers fork guard ----------------------------------------------
# Silences the fast-tokenizers fork warning and avoids the rare deadlock when
# tokenizers' Rust threadpool is forked into vLLM worker processes.
if [[ "${HF_ENABLE_TOKENIZERS_GUARD}" == "1" && -z "${TOKENIZERS_PARALLELISM:-}" ]]; then
  export TOKENIZERS_PARALLELISM=false
  hf_opt_log "TOKENIZERS_PARALLELISM=false"
fi
